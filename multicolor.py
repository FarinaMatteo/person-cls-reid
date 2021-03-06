"""Define functions to add the 'upmulticolor' and 'downmulticolor' columns in the annotations DataFrame."""

def get_down(row):
    """Get the value of 'downmulticolor' based on an annotation :row:"""
    check = True
    for k,v in row.items():
        if k.startswith("down") and k!="down" and v==2:
            check = False
    return check

def get_up(row):
    """Get the value of 'upmulticolor' based on an annotation :row:"""
    check = True
    for k,v in row.items():
        if k.startswith("up") and k!="up" and v==2:
            check = False
    return check

def multi_color(df):
    """Insert the values of 'upmulticolor' and 'downmulticolor' columns in a Dataframe."""
    up = []
    down = []
    for i in range(len(df)):
        a = df.iloc[i].to_dict()
        c_up = get_up(a)
        c_down = get_down(a)
        if c_up==True:
            up.append(2)
        else:
            up.append(1)
        if c_down==True:
            down.append(2)   
        else:
            down.append(1)
    
    df['upmulticolor'] = up
    df['downmulticolor'] = down

    df = df[["id", "age", "backpack", "bag", "handbag", "clothes", "down", "up", "hair", "hat", "gender",\
            "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen", "upmulticolor", \
            "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen", "downbrown", "downmulticolor"]]

    return df
