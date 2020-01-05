
class NNResult:
    def __init__(self):
        self.steering = 0
        self.shifting = 0
        self.wrong = 0

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def print_info(self):
        label = ""
        label += 'Steering: ' + str(self.steering) + " %\t"
        label += 'Shifting: ' + str(self.shifting) + " %\t"
        label += 'Wrong: ' + str(self.wrong) + " %\t"
        print(label)

    def process_result(self):
        maximum = 0
        className = ""
        color = (0, 0, 255)
        for attr, value in self.__dict__.items():
            if(value > maximum):
                className = attr
                maximum = value
                color = self.get_class_color(className)
        text = className + ": " + str(maximum) + " %"
        return text, color, className

    def get_class_color(self, className):
        if className == "steering":
            return (0, 255, 0)
        elif className == "shifting":
            return (0, 127, 255)
        else:
            return (0, 0, 255)
