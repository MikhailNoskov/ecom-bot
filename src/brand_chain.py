import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app_lc import CliBot
from schemas import ReplySchema, UsageSchema


class DemoBot(CliBot):
    def __init__(self, session_id, question, *args, **kwargs):
        super().__init__(session_id)
        self.__user_question = question
        self.output_parser = JsonOutputParser(pydantic_object=ReplySchema)
        # self.chain_with_history = self.chain_with_history | self.output_parser

    def __call__(self, *args, **kwargs):
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

    # def _update_system_prompt(self):
    #     faq_data_escaped = json.dumps(self._faq, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')
    #     orders_data_escaped = json.dumps(self._orders, ensure_ascii=False, indent=2).replace('{', '{{').replace('}',
    #                                                                                                                '}}')
    #     system_prompt = f'''
    #         {self._style.tone.role} бренда {self._style.brand}. Всегда сохраняй стиль ответа: {self._style.tone.persona}.
    #         Отвечай не более {self._style.tone.sentences_max} предложений, избегая {self._style.tone.avoid} и включая
    #         {self._style.tone.must_include}. Отвечай на {self._style.task}, используя правила: {self._style.rules}.
    #         Ориентируйся при ответах на часто задаваемые вопросы {faq_data_escaped}, а при вопросах о заказах
    #         ориентируйся на данные по заказам из {orders_data_escaped}
    #         При ответе используй чёткий формат ответа: {self._style.format.fields}
    #         '''
    #
    #     return ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{question}"),
    #     ])


def ask(question: str) -> ReplySchema:
    bot = DemoBot(111, question)
    return bot()


if __name__ == "__main__":
    data = input("Input your test question: ")
    print(ask(data))