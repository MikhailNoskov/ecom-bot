from app_lc import CliBot
from schemas import ReplySchema

class DemoBot(CliBot):
    def __init__(self, session_id, question, *args, **kwargs):
        super().__init__(session_id)
        self.__user_question = question

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
        bot_reply = response.content.strip()
        response_meta = response.response_metadata.get("token_usage", None)
        if response_meta is not None:
            del response_meta["completion_tokens_details"]
            del response_meta["prompt_tokens_details"]
        return ReplySchema(**bot_reply)


def ask(question: str) -> ReplySchema:
    bot = DemoBot(111, question)
    return bot()


if __name__ == "__main__":
    data = input("Input your test question: ")
    print(ask(data))