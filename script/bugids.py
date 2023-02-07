import os

projs = ["Cli", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath"]
bugs = [39, 18, 4, 47, 16, 18, 26, 112, 6, 93, 22]
with open("data/bugid2.txt", "w") as f:
    for proj, bug in zip(projs, bugs):
        for i in range(1, bug + 1):
            f.write(f"{proj}-{i}\n")