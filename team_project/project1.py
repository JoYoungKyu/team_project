from roboflow import Roboflow
rf = Roboflow(api_key="tEFOW2xP2wtvNyZmpXnL")
project = rf.workspace("jyk-ucnhk").project("jyk")
version = project.version(7)
dataset = version.download("yolov11")
                