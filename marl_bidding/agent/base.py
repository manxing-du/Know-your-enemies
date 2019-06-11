class BaseAgent:
    def act(self, o):
        raise NotImplementedError

    def act_om(self, o, market):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def final_sess_save(self):
        raise NotImplementedError
