
nn_count_interval = 4
nn_counter = 0
prev_text = None
prev_color = None
prev_className = None


def nn_filter(text, color, className, printInfo):
    global nn_count_interval
    global nn_counter
    global prev_text
    global prev_color
    global prev_className

    if(prev_className is None or className == prev_className):
        nn_counter = 0
        prev_className = className
        prev_text = text
        prev_color = color
    else:
        nn_counter += 1
        if printInfo is True:
            print("NN Class changed. Applying filter:  " + str(nn_counter) + "/" + str(nn_count_interval))
        if nn_counter > nn_count_interval:
            nn_counter = 0
            prev_className = className
            prev_text = text
            prev_color = color
    return prev_text, prev_color, prev_className
