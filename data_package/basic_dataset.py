import os
from torch.utils import data


class BasicDataset(data.Dataset):
    def __init__(self, root_dirs,
                 data_folder_name = 'data',data_extensions=None,
                 label_folder_name='label',label_extensions=None,
                 transform=None
                 ):
        super(BasicDataset, self).__init__()
        self.root_dirs=root_dirs if isinstance(root_dirs,list) else [root_dirs]
        self.data_folder_name = data_folder_name
        self.label_folder_name = label_folder_name
        self.transform=transform

        if not data_extensions == None:
            self.data_extensions = data_extensions if isinstance(data_extensions,list) else [data_extensions]
        else:
            self.data_extensions = None
        if not label_extensions == None:
            self.label_extensions = label_extensions if isinstance(label_extensions,list) else [label_extensions]
        else:
            self.label_extensions=['txt']
        self.params={}
        self.add_params()
        self.file_list= self._get_index()

    def _get_index(self):
        file_list=[]
        for root_dir in self.root_dirs:
            data_dir_path = os.path.join(root_dir,self.data_folder_name)
            data_dir = [x for x in os.walk(data_dir_path)]
            data_dir = data_dir[0]
            if len(data_dir[2])>0:
                for file_idx,data_file_name in enumerate(data_dir[2]):
                    data_file_name_split =data_file_name.split('.')
                    data_file_name ='.'.join(data_file_name_split[:-1])
                    data_extension = data_file_name_split[-1]
                    if not self.data_extensions == None:
                        if not data_extension in self.data_extensions:
                            continue
                    data_file_path = os.path.join(root_dir,self.data_folder_name,data_file_name+'.'+data_extension)

                    label_file_path = None
                    for label_extension in self.label_extensions:
                        cur_label_file_path = os.path.join(root_dir,self.label_folder_name,data_file_name+'.'+label_extension)
                        if os.path.exists(cur_label_file_path):
                            label_file_path = cur_label_file_path
                            break
                    if label_file_path is None:
                        print("|WAR:The corresponding label file doesn't exist|")
                        continue
                    else:
                        with open(label_file_path,"r") as f:
                            readlines = f.readlines()
                            if readlines[0] == "T":
                                label = 1
                            elif readlines[0]=="F":
                                label = 0
                            else:
                                raise Exception("|ERR:wrong|")
                        file_list.append([data_file_path,label])
                    if file_idx %1000==999:
                        print("|INFO: 已添加文件%d|" % len(file_list))
            else:
                raise Exception("|ERR: 包含空文件夹|")
        print("|INFO: 已添加文件%d|" % len(file_list))
        return file_list

    def add_params(self):
        pass

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    BasicDataset(root_dirs='D:\data\smoke_car\\rebuild_data_slim\\base_dataset\MLP_dataa',
                 data_folder_name='data',
                 data_extensions='jpg',
                 label_folder_name='label',
                 label_extensions='txt'
                 )