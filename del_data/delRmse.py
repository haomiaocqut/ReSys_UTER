with open("10-10Baby.txt") as uif:
    rmstr = ""
    num =0
    line = uif.readline()
    while line is not None and line != "":
        par = line.partition("rmse = ")
        parh = par[2]
        parq = parh.partition(",")[0]
        rmse = float(parq)
        rmse = round(rmse,3)
        if num <= 200:
            print(rmse)
            num += 1
        line = uif.readline()

'''
    if num <= 200:
            rmstr = rmstr + str(rmse) + " "
            num += 1
        line = uif.readline()
    print(rmstr.strip(" "))
'''
