
nn_count_interval = 10
nn_counter = 0
prev_text = None
prev_color = None
prev_className = None


def nn_filter(text, color, className):
    global nn_count_interval
    global nn_counter
    global prev_text
    global prev_color
    global prev_className

    if(prev_className == None or className == prev_className):
        nn_counter = 0
        prev_className = className
        prev_text = text
        prev_color = color
    else:
        nn_counter += 1
        print("ERROR: " +  str(nn_counter))
        if(nn_counter > nn_count_interval):
            nn_counter = 0
            prev_className = className
            prev_text = text
            prev_color = color
    return prev_text, prev_color, prev_className
