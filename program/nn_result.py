
class NNResult:
    def __init__(self):
        self.steering = 0
        self.shifting = 0
        self.wrong = 0

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def print_info(self):
        label = ""
        label += 'Steering: ' + str(round(self.steering)) + " %\t"
        label += 'Shifting: ' + str(round(self.shifting)) + " %\t"
        label += 'Wrong: ' + str(round(self.wrong)) + " %\t"
        print(label)

    def process_result(self):
        maximum = 0
        class_name = ""
        color = (0, 0, 255)
        for attr, value in self.__dict__.items():
            if(value > maximum):
                class_name = attr
                maximum = value
                color = self.get_class_color(class_name)
        text = class_name + ": " + str(round(maximum, 2)) + " %"
        return text, color, class_name

    def get_class_color(self, class_name):
        if class_name == "steering":
            return (0, 255, 0)
        elif class_name == "shifting":
            return (0, 127, 255)
        else:
            return (0, 0, 255)

class NN_result_counter:
    def __init__(self):
        self.steering = 0
        self.shifting = 0
        self.wrong = 0

    def increment(self, class_name):
        if class_name is not None:
            temp = getattr(self, class_name) + 1
            setattr(self, class_name, temp)
