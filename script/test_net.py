class T(object):
    def __init__(self):
        self.a = 10

    def setB(self, b):
        self.b = b

    def getB(self):
        print self.b


if __name__ == "__main__":
    t = T()
    t.setB(3)
    t.getB()
