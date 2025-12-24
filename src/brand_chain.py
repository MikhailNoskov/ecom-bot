import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app_lc import CliBot
from schemas import ReplySchema


class DemoBot(CliBot):
    def __init__(self, session_id, question, *args, **kwargs):
        super().__init__(session_id)
        self.__user_question = question
        self.output_parser = JsonOutputParser(pydantic_object=ReplySchema)
        self.chain_with_history = self.chain_with_history | self.output_parser

    def __call__(self, *args, **kwargs):
        msg = self.__user_question.lower()
        if msg in ("выход", "стоп", "конец"):
            return ReplySchema(answer="Бот: До свидания!")
        if msg in ("сброс", "обновить"):
            return ReplySchema(answer="Бот: Контекст диалога очищен.")
        faq_matching_entry = next((item for item in self._faq if item["q"] == self.__user_question), None)
        if faq_matching_entry:
            return ReplySchema(answer=f"Бот: {faq_matching_entry.get('a', None)}")
        if self.__user_question.startswith("/orders"):
            try:
                order_number = self.__user_question.split(" ")[1]
                if order_number.isdigit() and order_number in self._orders:
                    return ReplySchema(answer=f"Бот: {self._orders.get(order_number, None)}")
                else:
                    return ReplySchema(answer=f"Бот: No order info")
            except KeyError as err:
                return ReplySchema(answer=str(err))
            except Exception as err:
                return ReplySchema(answer=str(err))
        try:
            response = self.chain_with_history.invoke(
                {"question": self.__user_question},
                {"configurable": {"session_id": self.session_uuid}}
            )
        except Exception as err:
            return ReplySchema(answer=str(err))
        return response

    def _update_system_prompt(self):
        faq_data_escaped = json.dumps(self._faq, ensure_ascii=False, indent=2).replace('{', '{{').replace('}', '}}')
        orders_data_escaped = json.dumps(self._orders, ensure_ascii=False, indent=2).replace('{', '{{').replace('}',
                                                                                                                   '}}')
        system_prompt = f'''
            {self._style.tone.role} бренда {self._style.brand}. Всегда сохраняй стиль ответа: {self._style.tone.persona}.
            Отвечай не более {self._style.tone.sentences_max} предложений, избегая {self._style.tone.avoid} и включая
            {self._style.tone.must_include}. Отвечай на {self._style.task}, используя правила: {self._style.rules}.
            Ориентируйся при ответах на часто задаваемые вопросы {faq_data_escaped}, а при вопросах о заказах
            ориентируйся на данные по заказам из {orders_data_escaped}
            При ответе используй чёткий формат ответа: {self._style.format.fields}
            '''

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])


def ask(question: str) -> ReplySchema:
    bot = DemoBot(111, question)
    return bot()


if __name__ == "__main__":
    data = input("Input your test question: ")
    print(ask(data))