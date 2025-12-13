import json
import os
import sys
import logging
import uuid
from pythonjsonlogger.json import JsonFormatter
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


class UnicodeJsonFormatter(JsonFormatter):
    def jsonify_log_record(self, log_record):
        return json.dumps(
            log_record,
            ensure_ascii=False,
            separators=(",", ":")
        )


def setup_jsonl_logger(session_id: uuid.UUID):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/session_{session_id}.jsonl"
    logger = logging.getLogger("CliBot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = UnicodeJsonFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s %(event)s %(usage)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        rename_fields={"asctime": "timestamp", "levelname": "level"}
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class CliBot:
    def __init__(self, session_id):
        self.chat_model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-5"),
            temperature=0,
            timeout=15
        )
        self.session_uuid = session_id
        self.history_store = {}
        self.__faq = self.__get_faq_info()
        self.__orders = self.__get_orders_info()
        self.prompt = self.__update_system_prompt()
        self.chain = self.prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        self.logger = setup_jsonl_logger(self.session_uuid)

    def __update_system_prompt(self):
        faq_data_escaped = json.dumps(self.__faq, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')
        orders_data_escaped = json.dumps(self.__orders, ensure_ascii=False, indent=2).replace('{', '{{').replace('}',
                                                                                                                   '}}')
        system_prompt = f'''
            Ты полезный ассистент интернет магазина. Ты всегда дружелёбен и вежлив. Отвечай иеформативно, но сжато.
            Ориентируйся при ответах на часто задаваемые вопросы {faq_data_escaped}, а при получении команды вида /orders 00000
            ориентируйся на данные по заказам из {orders_data_escaped}
            '''

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def __call__(self):
        self.logger.info({"event": "session_start"})
        print(
            f"Чат-бот запущен {os.getenv('BRAND_NAME', 'No name')}! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        self.logger.info(f"=== New session {self.session_uuid}===")
        while True:
            user_text = None
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            except UnicodeDecodeError as err:
                self.logger.error(
                    err,
                    extra={"event": "error"})
                pass
            if not user_text:
                continue
            self.logger.info(
                user_text,
                extra={"event": "user input"}
            )
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logging.info(
                    "Бот: До свидания!",
                    extra={"event": "bot reply"}
                )
                logging.info({"event": "User session end"})
                break
            if msg in ("сброс", "обновить"):
                if self.session_uuid in self.history_store:
                    self.__init__(self.session_uuid)
                print("Бот: Контекст диалога очищен.")
                self.logger.info("dialog context cleared")
                continue
            faq_matching_entry = next((item for item in self.__faq if item["q"] == user_text), None)
            if faq_matching_entry:
                self.logger.info(
                    faq_matching_entry.get("a", None),
                    extra={"event": "faq reply"})
                print(f"Бот: {faq_matching_entry.get('a', None)}")
                continue
            if user_text.startswith("/orders"):
                try:
                    order_number = user_text.split(" ")[1]
                    if order_number.isdigit() and order_number in self.__orders:
                        self.logger.info(
                            self.__orders.get(order_number, None),
                            extra={"event": "faq reply"})
                        print(f"Бот: {self.__orders.get(order_number, None)}")
                    else:
                        self.logger.info(
                            "No order info",
                            extra={"event": "faq reply"})
                        print("Бот: No order info")
                except KeyError as err:
                    self.logger.error(
                        err,
                        extra={"event": "error"})
                except Exception as err:
                    self.logger.error(
                        err,
                        extra={"event": "error"})
                    print(f"[Ошибка] {err}")
                finally:
                    continue
            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": self.session_uuid}}
                )
            except Exception as err:
                self.logger.error(
                    err,
                    extra={"event": "error"})
                print(f"[Ошибка] {err}")
                continue
            bot_reply = response.content.strip()
            response_meta = response.response_metadata.get("token_usage", None)
            if response_meta is not None:
                del response_meta["completion_tokens_details"]
                del response_meta["prompt_tokens_details"]
            self.logger.info(
                bot_reply,
                extra={"event": "bot reply", "usage": response_meta})
            print(f"Бот: {bot_reply}")

    @classmethod
    def __get_faq_info(cls):
        with open('data/faq.json', 'r', encoding='utf-8') as f:
            faq_data_obj = json.load(f)
        return faq_data_obj

    @classmethod
    def __get_orders_info(cls):
        with open('data/orders.json', 'r', encoding='utf-8') as f:
            orders_data_obj = json.load(f)
        return orders_data_obj


def main():
    session_uuid = uuid.uuid4()
    CliBot(session_uuid)()

if __name__ == "__main__":
    sys.exit(main())
