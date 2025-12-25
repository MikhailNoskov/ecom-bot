import uuid

from langchain_core.output_parsers import JsonOutputParser

from app_lc import CliBot
from schemas import ReplySchema, UsageSchema


class DemoBot(CliBot):
    """Демонстрационная версия CLI-бота для однократного вызова без интерактивного цикла.

    Наследуется от `CliBot`, но вместо ожидания пользовательского ввода в цикле
    обрабатывает единственный предопределённый вопрос и сразу возвращает структурированный ответ.
    Используется в тестировании и автоматической оценке (`eval_batch`).

    Все ветки логики (FAQ, заказы, команды) сохранены, но исторический контекст не обновляется,
    а метрики использования токенов для не-LLM ответов подставляются как нулевые.
    """
    def __init__(self, session_id, question, *args, **kwargs):
        """Инициализирует демонстрационного бота с фиксированным вопросом.

        Args:
            session_id: Идентификатор сессии (может быть произвольным, например, 111).
            question (str): Вопрос, который бот должен обработать без интерактивного ввода.
        """
        super().__init__(session_id)
        self.__user_question = question
        self.output_parser = JsonOutputParser(pydantic_object=ReplySchema)

    def __call__(self, *args, **kwargs):
        """Обрабатывает сохранённый вопрос и возвращает ответ в виде ReplySchema.

        Поддерживает те же команды и логику, что и интерактивный `CliBot`:
        - завершение ('выход', 'стоп', 'конец'),
        - сброс контекста ('сброс', 'обновить'),
        - точное совпадение с FAQ,
        - запросы к заказам через '/orders <номер>',
        - и, в остальных случаях, генерацию через LLM.

        Для всех путей, не использующих LLM, подставляется нулевая метрика использования токенов.

        Returns:
            ReplySchema: Ответ с полями `answer`, `tone`, `actions`, `usage`.
        """
        msg = self.__user_question.lower()
        empty_usage = UsageSchema(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        if msg in ("выход", "стоп", "конец"):
            return ReplySchema(
                answer="Бот: До свидания!",
                usage=empty_usage
            )
        if msg in ("сброс", "обновить"):
            return ReplySchema(
                answer="Бот: Контекст диалога очищен.",
                usage=empty_usage
            )
        faq_matching_entry = next((item for item in self._faq if item["q"] == self.__user_question), None)
        if faq_matching_entry:
            return ReplySchema(
                answer=f"Бот: {faq_matching_entry.get('a', None)}",
                usage=empty_usage
            )
        if self.__user_question.startswith("/orders"):
            try:
                order_number = self.__user_question.split(" ")[1]
                if order_number.isdigit() and order_number in self._orders:
                    return ReplySchema(answer=f"Бот: {self._orders.get(order_number, None)}", usage=empty_usage)
                else:
                    return ReplySchema(answer=f"Бот: No order info", usage=empty_usage)
            except KeyError as err:
                return ReplySchema(answer=str(err), usage=empty_usage)
            except Exception as err:
                return ReplySchema(answer=str(err), usage=empty_usage)
        try:
            response = self._get_llm_reply(self.__user_question)
        except Exception as err:
            return ReplySchema(answer=str(err), usage=empty_usage)
        return response


def ask(question: str) -> ReplySchema:
    """Упрощённый интерфейс для получения ответа от бота на один вопрос.

    Создаёт временный экземпляр `DemoBot` с фиксированным session_id
    и возвращает результат обработки заданного вопроса.

    Args:
        question (str): Входной вопрос пользователя.

    Returns:
        ReplySchema: Строго типизированный ответ от бота.
    """
    bot = DemoBot(uuid.UUID('441c8040-ad85-4022-9acc-8b02405b3930'), question)
    return bot()


if __name__ == "__main__":
    data = input("Input your test question: ")
    print(ask(data))