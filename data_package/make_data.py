import cv2
import os

def get_file_name(path):
    file_names = [x for x in os.walk(path)][0][2]

    file_names = ['.'.join(x.split('.')[:-1]) for x in file_names]
    return file_names

def load_data(data_path:str,label_path:str):
    cap = cv2.VideoCapture(data_path)
    data_list = []
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        data_list.append(cv2.resize(frame,(300,300)))
    save_shape=[300,300]
    with open(label_path, "r", encoding="utf-8") as f:
        readlines = f.readlines()
        boxs = []
        for readline in readlines:
            readline_split = readline.strip().split(",")
            box = [
                [int(float(readline_split[0]) * save_shape[1]), int(float(readline_split[1]) * save_shape[0])],
                [int(float(readline_split[2]) * save_shape[1]), int(float(readline_split[3]) * save_shape[0])],
                readline_split[4]]
            boxs.append(box)
    return data_list,boxs


if __name__ == '__main__':
    import cv2
    import numpy as np

    org_data_dirs=[
        # "D:\data\smoke_car\\rebuild_data_slim\\base_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\DeAn_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\GuRun_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\HeNeng_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\TongHua_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\WanZai_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\XinXiang_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\YunJing_dataset",
        # "D:\data\smoke_car\\rebuild_data_slim\\ZhangYe_dataset",
        "D:\data\smoke_car\\rebuild_data_slim\\test_dataset",


                   ]
    save_data_pardir="D:\data\smoke_car/smoke_classification_data"

    for org_data_dir in org_data_dirs:
        save_data_dir = save_data_pardir+"/%s"%os.path.basename(org_data_dir)
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        file_names = get_file_name(org_data_dir+"/data")
        for file_name in file_names:
            data, label = load_data(
                "%s/data/%s.mp4"%(org_data_dir,file_name),
                "%s/label/%s.txt"%(org_data_dir,file_name))
            have_smoke=False
            for x in label:
                if x[-1] == "smoke":
                    have_smoke=True
                    label = x
                    break
                elif "smoke" in x[-1]:
                    print("")
                    continue
            if not have_smoke:
                continue
                center_point = np.random.randint(0, 300, 2)
                max_wh = 50
            else:
                # continue
                center_point = (np.array(label[0])+np.array(label[1]))//2
                max_wh = np.max(np.array(label[1])-np.array(label[0]))//2*3
                max_wh = min(80,max_wh)
            center_point=np.clip(center_point,max_wh,300-max_wh)
            video_data=[]
            for d in data:
                show_data = d[center_point[1]-max_wh:center_point[1]+max_wh,center_point[0]-max_wh:center_point[0]+max_wh,:]
                show_data = cv2.resize(show_data,(100,100))
                video_data.append(show_data)
            video_data = np.concatenate(video_data,0)
            save_path = "%s/data/%s.jpg"%(save_data_dir,file_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            cv2.imencode(".jpg",video_data)[1].tofile(save_path)
            save_path = "%s/label/%s.txt"%(save_data_dir,file_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(save_path,"w",encoding="utf-8") as f:
                if have_smoke:
                    f.write("T")
                else:
                    f.write("F")


