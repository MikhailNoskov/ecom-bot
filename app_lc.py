import json
from typing import List, Dict

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
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
    """Форматтер логов в формате JSON с поддержкой Unicode-символов без экранирования.

    Наследуется от JsonFormatter и переопределяет метод сериализации,
    чтобы обеспечить читаемость не-ASCII символов (например, кириллицы)
    в логах за счёт параметра `ensure_ascii=False`.
    """
    def jsonify_log_record(self, log_record):
        """Сериализует запись лога в компактную JSON-строку с поддержкой Unicode.

        Args:
            log_record (dict): Словарь с полями записи лога.

        Returns:
            str: JSON-строка без экранирования Unicode-символов и с минимальными разделителями.
        """
        return json.dumps(
            log_record,
            ensure_ascii=False,
            separators=(",", ":")
        )

def setup_jsonl_logger(session_id: uuid.UUID):
    """Настраивает и возвращает JSONL-логгер для заданной сессии.

    Создаёт директорию `logs/`, если она отсутствует, и инициализирует
    файловый обработчик, записывающий логи в формате JSON Lines (по одной записи на строку).
    Каждая запись содержит временну́ю метку, уровень, сообщение, тип события и данные об использовании.

    Args:
        session_id (uuid.UUID): Уникальный идентификатор сессии для имени файла лога.

    Returns:
        logging.Logger: Настроенный экземпляр логгера с именем "CliBot".
    """
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
    """Клиентский чат-бот для интерактивного взаимодействия с пользователем через терминал.

    Класс реализует CLI-интерфейс к LLM-системе с поддержкой:
    - хранения истории диалога по сессиям,
    - семантического поиска в FAQ и данных о заказах,
    - few-shot контекстуального обучения,
    - логирования событий в формате JSONL,
    - строгой схемы вывода (ReplySchema).

    При инициализации загружает стилистику, примеры, FAQ и информацию о заказах,
    настраивает цепочку вызова LLM и подготовку контекста.
    """
    def __init__(self, session_id: uuid.UUID):
        """Инициализирует экземпляр чат-бота для заданной сессии.

        Args:
            session_id (str): Уникальный идентификатор пользовательской сессии.
        """
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
        self.faq_store = Chroma.from_documents(self._get_faq_docs(), self.embeddings)
        self.orders_store = Chroma.from_documents(self._get_orders_docs(), self.embeddings)
        self._faq = self._get_faq_info()
        self._orders = self._get_orders_info()
        self.prompt = self._update_system_prompt()
        self.chain = self.prompt | self.chat_model
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        self.logger = setup_jsonl_logger(self.session_uuid)

    @classmethod
    def _get_style(cls):
        """Загружает стилистические настройки из глобальной константы STYLE.

        Returns:
            StyleSchema: Объект, содержащий правила тона, бренда, формата и содержания ответов.
        """
        return StyleSchema(**STYLE)

    def _update_system_prompt(self) -> ChatPromptTemplate:
        """Формирует системный промпт с учётом стиля, FAQ, заказов и few-shot примеров.

        Returns:
            ChatPromptTemplate: Шаблон промпта для LLM с плейсхолдерами истории и контекста.
        """
        system_prompt = f'''
            {self._style.tone.role} - {self._style.brand}. Always keep the reply style: {self._style.tone.persona}.
            Not more than {self._style.tone.sentences_max} sentences, avoiding {self._style.tone.avoid} but including
            {self._style.tone.must_include}. Reply to {self._style.task}, using the rules: {self._style.rules} strictly 
            keeping the reply schema format{self._style.format}.
            Take into account the context of faq from {{faqs}} and of order info from {{orders}}
            '''
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("system", "{few_shots}"),
            ("system", "{faqs}"),
            ("system", "{orders}"),
            ("human", "{question}"),
        ])

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Возвращает или создаёт историю сообщений для заданной сессии.

        Args:
            session_id (str): Идентификатор сессии.

        Returns:
            InMemoryChatMessageHistory: Объект истории сообщений для данной сессии.
        """
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def _render_few_shots(self, question: str) -> str:
        """Генерирует строковое представление релевантных few-shot примеров для вопроса.

        Args:
            question (str): Вопрос пользователя.

        Returns:
            str: Текст в формате «User: ...\nAssistant: ...», разделённый двойными переносами.
        """
        examples = self.example_selector.select_examples({"question": question})
        return "\n\n".join(
            f"User: {ex['question']}\nAssistant: {ex['answer']}" if len(ex) > 0 else ""
            for ex in examples
        )

    def _retrieve_faq(self, question: str) -> str:
        """Извлекает релевантные записи из базы FAQ по семантическому сходству.

        Args:
            question (str): Входной вопрос.

        Returns:
            str: Объединённое содержимое до 3 наиболее релевантных FAQ-документов.
        """
        docs = self.faq_store.similarity_search(question, k=3)
        return "\n\n".join(d.page_content for d in docs)

    def _retrieve_orders(self, question: str) -> str:
        """Извлекает информацию о заказах, релевантную вопросу (обычно по номеру).

        Args:
            question (str): Вопрос, потенциально содержащий номер заказа.

        Returns:
            str: Содержимое наиболее релевантного документа с данными о заказе.
        """
        docs = self.orders_store.similarity_search(question, k=1)
        return "\n\n".join(d.page_content for d in docs)

    def _get_faq_docs(self) -> List[Document]:
        """Преобразует FAQ-данные в список объектов Document для векторного хранилища.

        Returns:
            list[Document]: Список документов в формате «Q: ...\nA: ...».
        """
        return [
            Document(page_content=f"Q: {item['q']}\nA: {item['a']}")
            for item in self._get_faq_info()
        ]

    def _get_orders_docs(self) -> List[Document]:
        """Преобразует данные о заказах в список объектов Document для векторного хранилища.

        Returns:
            list[Document]: Список документов в формате «ключ: значение».
        """
        return [
            Document(page_content=f"{k}: {v}")
            for k, v in self._get_orders_info().items()
        ]

    @classmethod
    def _get_faq_info(cls) -> List[Dict]:
        """Загружает FAQ из JSON-файла.

        Returns:
            list[dict]: Список словарей с ключами 'q' (вопрос) и 'a' (ответ).
        """
        with open('data/faq.json', 'r', encoding='utf-8') as file:
            faq_data_obj = json.load(file)
        return faq_data_obj

    @classmethod
    def _get_orders_info(cls) -> Dict[int, Dict]:
        """Загружает информацию о заказах из JSON-файла.

        Returns:
            dict: Словарь, где ключи — номера заказов, значения — детали заказа.
        """
        with open('data/orders.json', 'r', encoding='utf-8') as file:
            orders_data_obj = json.load(file)
        return orders_data_obj

    @classmethod
    def _get_samples(cls) -> List[Dict]:
        """Загружает few-shot примеры из JSONL-файла.

        Returns:
            list[dict]: Список словарей с парами 'question' и 'answer'.
        """
        with open('data/few_shots.jsonl', 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]

    def _get_llm_reply(self, text: str) -> ReplySchema:
        """Получает структурированный ответ от LLM с учётом контекста и ретривера.

        Args:
            text (str): Вопрос пользователя.

        Returns:
            ReplySchema: Строго типизированный ответ, включающий текст, тон и действия.
        """
        few_shots = self._render_few_shots(text)
        faqs = self._retrieve_faq(text)
        orders = self._retrieve_orders(text)
        return self.chain_with_history.invoke(
            {
                "question": text,
                "few_shots": few_shots,
                "faqs": faqs,
                "orders": orders
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
    ) -> None:
        """Выводит сообщение в консоль и записывает его в лог с метаданными.

        Args:
            event (str): Тип события (например, 'bot_reply', 'error').
            msg (str, optional): Сообщение для вывода и лога (при INFO).
            err_msg (str, optional): Описание ошибки (при ERROR).
            usage (dict, optional): Данные об использовании токенов LLM.
            log_level (int): Уровень логирования (INFO или ERROR).
        """
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

    def _process_user_input(self) -> str:
        """Обрабатывает ввод пользователя из стандартного потока ввода.

        Returns:
            str: Очищенный текст введённого пользователем сообщения.

        Raises:
            KeyboardInterrupt, EOFError: При завершении сессии (Ctrl+C, Ctrl+D).
            UnicodeDecodeError: При проблемах с кодировкой ввода.
        """
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

    def __call__(self):
        """Запускает основной цикл CLI-чата.

        Обрабатывает команды:
        - 'выход', 'стоп', 'конец' — завершение сессии,
        - 'сброс', 'обновить' — сброс контекста сессии,
        - точные совпадения с FAQ — мгновенный ответ без LLM,
        - команды вида '/orders <номер>' — поиск данных о заказе.

        В остальных случаях делегирует генерацию ответа LLM.
        """
        self.logger.info({"event": f"session_start: {self.session_uuid}"})
        print(
            f"Чат-бот запущен {os.getenv('BRAND_NAME', 'No name')}! Можете задавать вопросы. \n - Для выхода введите 'выход'.\n - Для очистки контекста введите 'сброс'.\n")
        self.logger.info(f"=== New session {self.session_uuid}===")
        while True:
            # Юзерский ввод
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
            # Ответ без llm
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
            # Ответ llm
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
                msg=f"Бот: {response.answer}\nTone: {response.tone}\nActions: {response.actions}\n",
                usage={
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )


def main():
    """Точка входа в CLI-приложение чат-бота.

    Генерирует уникальный UUID для новой пользовательской сессии
    и запускает интерактивный цикл чата через экземпляр CliBot.

    """
    session_uuid = uuid.uuid4()
    CliBot(session_uuid)()


if __name__ == "__main__":
    sys.exit(main())
