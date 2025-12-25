import json
import yaml
import os
import sys
import pathlib
import logging
import uuid
from pythonjsonlogger.json import JsonFormatter
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, FewShotPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from schemas import StyleSchema, ReplySchema

__import__('pysqlite3')

load_dotenv()
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

BASE = pathlib.Path(__file__).parent

with open("data/style_guide.yaml", "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)

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
        print("Initializing Bot")
        self.chat_model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-5"),
            temperature=0,
            timeout=15
        ).with_structured_output(ReplySchema)
        self.session_uuid = session_id
        self.history_store = {}
        self._style = self._get_style()

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=self._get_samples(),
            embeddings=self.embeddings,
            vectorstore_cls=Chroma,
            k=2
        )
        self.example_prompt = PromptTemplate.from_template("Вопрос: {question}\nОтвет: {answer}")
        # self.faq_store = Chroma.from_documents(self._get_faq_docs(), self.embeddings)
        # self.orders_store = Chroma.from_documents(self._get_orders_docs(), self.embeddings)
        self._faq = self._get_faq_info()
        self._orders = self._get_orders_info()
        self.prompt = self._update_system_prompt()
        self.chain = self.prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        self.logger = setup_jsonl_logger(self.session_uuid)
        print("Bot initialized")

    @classmethod
    def _get_style(cls):
        return StyleSchema(**STYLE)

    def _update_system_prompt(self):
        faq_data_escaped = json.dumps(self._faq, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')
        orders_data_escaped = json.dumps(self._orders, ensure_ascii=False, indent=2).replace('{', '{{').replace('}',
                                                                                                                   '}}')
        system_prompt = f'''
            {self._style.tone.role} - {self._style.brand}. Always keep the reply style: {self._style.tone.persona}.
            Not more than {self._style.tone.sentences_max} sentences, avoiding {self._style.tone.avoid} but including
            {self._style.tone.must_include}. Reply to {self._style.task}, using the rules: {self._style.rules} strictly 
            keeping the reply schema format{self._style.format}.
            Take into account the context of faq from {faq_data_escaped} and of order info from {orders_data_escaped}
            '''

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("system", "{few_shots}"),
            ("human", "{question}"),
        ])

    # def _update_system_prompt(self):
    #     return ChatPromptTemplate.from_messages([
    #         ("system", f"""
    #         Ты {self._style.tone.role} бренда {self._style.brand}.
    #         Всегда соблюдай стиль: {self._style.tone.persona}.
    #         Максимум {self._style.tone.sentences_max} предложений.
    #         Запрещено: {self._style.tone.avoid}.
    #         Обязательно: {self._style.tone.must_include}.
    #         Отвечай, используя правила: {self._style.rules} и структуру {ReplySchema}."""),
    #         ("system", "Примеры правильных ответов:\n{few_shots}"),
    #         ("system", "Часто задаваемые вопросы:\n{faq_context}"),
    #         ("system", "Информация по заказам:\n{orders_context}"),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{question}"),
    #     ])

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def __call__(self):
        self.logger.info({"event": f"session_start: {self.session_uuid}"})
        print(
            f"Чат-бот запущен {os.getenv('BRAND_NAME', 'No name')}! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        self.logger.info(f"=== New session {self.session_uuid}===")
        while True:
            #User input
            try:
                user_text = self._process_user_input()
                if not user_text:
                    continue
            except (KeyboardInterrupt, EOFError):
                break
            except UnicodeDecodeError:
                continue
            except Exception as err:
                self._chat_and_log(
                    event="error",
                    msg=f"\nБот: Завершение работы из-за ошибки {str(err)}",
                    err_msg=str(err),
                    log_level=logging.ERROR
                )
                break
            #Reply without model
            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                self._chat_and_log(
                    event="bot_reply",
                    msg="Бот: До свидания!"
                )
                self.logger.info({"event": "User session end"})
                break
            if msg in ("сброс", "обновить"):
                if self.session_uuid in self.history_store:
                    self.__init__(self.session_uuid)
                self._chat_and_log(
                    event="context_reset",
                    msg="Бот: Контекст диалога очищен."
                )
                continue
            faq_matching_entry = next((item for item in self._faq if item["q"] == user_text), None)
            if faq_matching_entry:
                self._chat_and_log(
                    event="faq_reply",
                    msg=f"Бот: {faq_matching_entry.get('a', None)}"
                )
                continue
            if user_text.startswith("/orders"):
                try:
                    order_number = user_text.split(" ")[1]
                    if order_number.isdigit() and order_number in self._orders:
                        self._chat_and_log(
                            event="order_reply",
                            msg=f"Бот: {self._orders.get(order_number, None)}"
                        )
                    else:
                        self._chat_and_log(
                            event="order_reply",
                            msg=f"Бот: No order info"
                        )
                except KeyError as err:
                    self._chat_and_log(
                        event="error",
                        err_msg=str(err),
                        log_level=logging.ERROR
                    )
                except Exception as err:
                    self._chat_and_log(
                        event="error",
                        err_msg=str(err),
                        log_level=logging.ERROR
                    )
                finally:
                    continue
            #Model reply
            try:
                response = self._get_llm_reply(user_text)
            except Exception as err:
                self._chat_and_log(
                    event="error",
                    msg="Бот: Возникла непредвиденная ошибка",
                    err_msg=str(err),
                    log_level=logging.ERROR
                )
                continue
            self._chat_and_log(
                event="bot_reply",
                msg=f"Бот: {response.answer}",
                usage={
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

    def _get_llm_reply(self, text: str) -> ReplySchema:
        few_shots = self._render_few_shots(text)
        return self.chain_with_history.invoke(
            {
                "question": text,
                "few_shots": few_shots,
            },
            {"configurable": {"session_id": self.session_uuid}},
        )

    def _chat_and_log(
            self,
            event: str,
            msg: str = None,
            err_msg: str = None,
            usage: dict = None,
            log_level=logging.INFO
    ):
        if log_level == logging.INFO:
            print(msg)
            self.logger.info(
                msg,
                extra={"event": f"{event}", "usage": usage}
            )
        else:
            print(msg)
            self.logger.error(
                err_msg,
                extra={"event": f"{event}"}
            )

    def _process_user_input(self):
        try:
            user_text = input("Вы: ").strip()
            self.logger.info(
                user_text,
                extra={"event": "user_input"}
            )
        except (KeyboardInterrupt, EOFError):
            self._chat_and_log(
                event="bot_reply",
                msg="\nБот: Завершение работы."
            )
            self.logger.info({"event": "User session end"})
            raise
        except UnicodeDecodeError as err:
            self._chat_and_log(
                event="bot error",
                msg="\nБот: Проблемы с кодировкой",
                err_msg=str(err),
                log_level=logging.ERROR
            )
            raise
        return user_text

    def _render_few_shots(self, question: str) -> str:
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt,
            prefix="",
            suffix="",
            input_variables=["question"],
        )
        return few_shot_prompt.format(question=question)
        # examples = self.example_selector.select_examples({"question": question})
        # print(examples)
        # return "\n\n".join(
        #     f"User: {ex['question']}\nAssistant: {ex['answer']}" if len(ex) > 0 else ""
        #     for ex in examples
        # )


    # def _retrieve_faq(self, question: str) -> str:
    #     docs = self.faq_store.similarity_search(question, k=3)
    #     return "\n\n".join(d.page_content for d in docs)
    #
    # def _retrieve_orders(self, question: str) -> str:
    #     docs = self.orders_store.similarity_search(question, k=1)
    #     return "\n\n".join(d.page_content for d in docs)

    def _get_faq_docs(self):
        return [
            Document(page_content=f"Q: {item['q']}\nA: {item['a']}")
            for item in self._get_faq_info()
        ]

    @classmethod
    def _get_faq_info(cls):
        with open('data/faq.json', 'r', encoding='utf-8') as f:
            faq_data_obj = json.load(f)
        return faq_data_obj

    def _get_orders_docs(self):
        return [
            Document(page_content=f"{k}: {v}")
            for k, v in self._get_orders_info().items()
        ]

    @classmethod
    def _get_orders_info(cls):
        with open('data/orders.json', 'r', encoding='utf-8') as f:
            orders_data_obj = json.load(f)
        return orders_data_obj

    @classmethod
    def _get_samples(cls):
        with open('data/few_shots.jsonl', 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]


def main():
    session_uuid = uuid.uuid4()
    CliBot(session_uuid)()


if __name__ == "__main__":
    sys.exit(main())
