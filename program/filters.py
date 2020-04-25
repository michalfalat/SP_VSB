
nn_threshold = 4
nn_counter = 0
prev_text = None
prev_color = None
prev_class_name = None


def nn_filter(text, color, class_name, printInfo):
    global nn_count_interval
    global nn_counter
    global prev_text
    global prev_color
    global prev_class_name

    if(prev_class_name is None or class_name == prev_class_name):
        nn_counter = 0
        prev_class_name = class_name
        prev_text = text
        prev_color = color
    else:
        nn_counter += 1
        if printInfo is True:
            print("NN Class changed. Applying filter:  " + str(nn_counter) + "/" + str(nn_threshold))
        if nn_counter > nn_threshold:
            nn_counter = 0
            prev_class_name = class_name
            prev_text = text
            prev_color = color
    return prev_text, prev_color, prev_class_name
