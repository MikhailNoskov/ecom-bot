import json
import os
import sys
import logging
import uuid

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


class CliBot:
    def __init__(self, session_id):
        self.chat_model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-5"),
            temperature=0,
            timeout=15
        )
        self.session_uuid = session_id
        self.history_store = {}
        self.prompt = self.__update_system_prompt()
        self.chain = self.prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    @classmethod
    def  __get_session_uuid(cls):
        return uuid.uuid4()

    @classmethod
    def __update_system_prompt(cls):
        with open('data/faq.json', 'r', encoding='utf-8') as f:
            faq_data_obj = json.load(f)

        with open('data/orders.json', 'r', encoding='utf-8') as f:
            orders_data_obj = json.load(f)

        faq_data_escaped = json.dumps(faq_data_obj, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')
        orders_data_escaped = json.dumps(orders_data_obj, ensure_ascii=False, indent=2).replace('{', '{{').replace('}',
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
        print(
            f"Чат-бот запущен {os.getenv('BRAND_NAME', 'No name')}! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        logging.info(f"=== New session {self.session_uuid}===")
        while True:
            user_text = None
            try:
                user_text = input("Вы: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            except UnicodeDecodeError as err:
                logging.error(err)
                pass
            if not user_text:
                continue

            logging.info(f"User: {user_text}")
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                logging.info("Пользователь завершил сессию. Сессия окончена.")
                break
            if msg in ("сброс", "обновить"):
                if self.session_uuid in self.history_store:
                    self.__update_system_prompt()
                    del self.history_store[self.session_uuid]
                print("Бот: Контекст диалога очищен.")
                logging.info("Пользователь сбросил контекст.")
                continue

            try:
                response = self.chain_with_history.invoke(
                    {"question": user_text},
                    {"configurable": {"session_id": self.session_uuid}}
                )
            except Exception as e:
                logging.error(f"[error] {e}")
                print(f"[Ошибка] {e}")
                continue

            bot_reply = response.content.strip()
            response_meta = response.response_metadata.get("token_usage", None)
            log_message = f"Bot: {bot_reply} "
            if response_meta is not None:
                del response_meta["completion_tokens_details"]
                del response_meta["prompt_tokens_details"]
                log_message += f"Usage: {response_meta}"
            logging.info(log_message)
            print(f"Бот: {bot_reply}")

if __name__ == "__main__":
    session_uuid = uuid.uuid4()
    logging.basicConfig(
        filename=f"logs/session_{session_uuid}.jsonl",
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s", datefmt = "%Y-%m-%d %H:%M:%S",
        encoding = "utf-8"
    )
    sys.exit(CliBot(session_uuid)())
