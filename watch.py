import os
import time
import glob
import subprocess

while True:
    l = glob.glob("{}_watch/*".format(os.getenv("NCCS_PRJ_PATH")))
    if len(l) > 0:
        for i in l:
            with open(i) as f:
                cmd = f.readlines()[0].split()
                executable = cmd[0]
                params = cmd[1:]
                cmds = ["python", os.getenv("local_prj_path",
                    "/PycharmProjects/RLDock/src") + "/" + executable] + params
                output = subprocess.check_output(cmds, shell=True,
                        stderr=subprocess.STDOUT)
                print(cmds, output)
                with open("{}/debug.log".format(os.getenv("NCCS_PRJ_PATH")), "a+") as fdebug:
                    fdebug.write("{}:{}\n".format(cmds, output))
            os.unlink("{}".format(i))
    else:
        time.sleep(5)
        print("wait")
