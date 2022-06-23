
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# importing necessary packages
import os
import ast
import re
from collections import Counter

# streamlit modules
import streamlit as st
# from streamlit_tensorboard import st_tensorboard 
from __init__ import st_tensorboard
# image operations modules
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# mathematical and data manipulation
import numpy as np
import pandas as pd

# yolov5 functions
from yolov5_files.detect import run

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# setting the page configurations
st.set_page_config(layout='wide')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# to calculate IOU score b/w two bbox values
def bb_intersection_over_union(box1, box2):
    """ We assume that the box follows the format:
    box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
    where (x1,y1) and (x3,y3) represent the top left coordinate,
    and (x2,y2) and (x4,y4) represent the bottom right coordinate """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_inter1 = max(x1, x3)
    y_inter1 = max(y1, y3)
    x_inter2 = min(x2, x4)
    y_inter2 = min(y2, y4)
    width_inter = abs(x_inter2 - x_inter1)
    height_inter = abs(y_inter2 - y_inter1)
    area_inter = width_inter * height_inter
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    area_union = area_box1 + area_box2 - area_inter
    iou = area_inter / area_union
    return iou

# distance calculation b/w two array of values
def distance(boxA, boxB):
    return(list(np.abs(np.array(boxA)- np.array(boxB))))


def convert_df(df):
    return df.to_csv().encode('utf-8')

# main function of streamlit
def main():
    st.title("AI-Inventory Management[V2]")
    st.write("You can view real-time object detection done using YOLO model here.")
    st.sidebar.image(Image.open('ubs.png'))

    uploaded_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpeg', 'jpg', 'JPG'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='loading...'):
            picture = Image.open(uploaded_file)
            # picture = picture.resize((5184,3888))
            st.sidebar.image(picture)
            picture1 = picture.save(f'yolov5_files/data1/images/{uploaded_file.name}')
            print('picture saved')
            model = 'v5'
            MODEL_WEIGHTS = [
                os.path.sep.join([os.getcwd(), "weights/yolov5_run17_mAP_87.pt"]),
                os.path.sep.join([os.getcwd(), "weights/yolov5_run22_mAP_88.pt"]),
                os.path.sep.join([os.getcwd(), "weights/yolov5_run54_mAP_91.pt"]),
                # os.path.sep.join([os.getcwd(), "weights/yolov5_run67_mAP_60.pt"]),
                os.path.sep.join([os.getcwd(), "weights/best.pt"]),
                os.path.sep.join([os.getcwd(), "weights/yolov5_run86_mAP_71.pt"])
                
            ]
            # print(MODEL_WEIGHTS)
    else:
        is_valid = False

            
    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)         

    if is_valid:
        print('valid')
        if st.button('detect'):
            col1, col2, col3, col4, col5 = st.columns(5)
            with st.spinner(text='loading...'):
                source = os.path.sep.join([os.getcwd(), f'yolov5_files/data1/images/{uploaded_file.name}'])
                # print(source)

                dict1={1:col1, 2:col2, 3:col3, 4:col4, 5:col5}
                count=0
                names = ['device', 'device', 'dtype', 'unused_rack']
                runs = []
                texts = []
                for model in range(len(MODEL_WEIGHTS)):
                    count+=1                    
                    img = run(weights=f"{MODEL_WEIGHTS[model]}", save_txt=f"{uploaded_file.name.split('.')[0]}.txt", conf_thres=score_threshold, source=source, device=1, save_conf=True)
                    
                    if MODEL_WEIGHTS[model].split('/')[-1] == 'yolov5_run86_mAP_71.pt':
                        model = 'ur'
                    elif MODEL_WEIGHTS[model].split('/')[-1] == 'best.pt':
                        model = 'ssr'
                    elif MODEL_WEIGHTS[model].split('/')[-1] == 'yolov5_run54_mAP_91.pt':
                        model = 'dri'
                        
                    print(img['text'])
                    if count != 1: 
                        dict1[count-1].write(f'M{count-2}_{names[count-2]}:')
                        dict1[count-1].image(img['im0'], use_column_width=True, channels='BGR')
                        texts.append(img['text'])

                    # st.image(img, channels='BGR', width=150, caption=MODEL_WEIGHTS[model].split('\\')[-1])

                    directory = os.path.sep.join([os.getcwd(),'yolov5_files/runs/detect'])
                    runs_print = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime)
                    runs.append(runs_print)
                    #
                    # for file in os.listdir(runs_print):
                    if os.path.exists(f"{runs_print}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                        print('file exists')
                        labels =[]
                        with open(f"{runs_print}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                            line = f.readlines()
                            for dt in line:
                                label, x, y, w, h, conf = dt.split(' ')
                                if model == 'ssr':
                                    label = 'server' if label == '0' else 'switch' if label == '1' else 'router'
                                elif model == 'ur':
                                    label = 'unused_rack_units'
                                elif model == 'dri':
                                    label = 'devices' if label == '0' else 'rackid'    
                                else:
                                    label = 'devices' if label == '0' else 'rack' if label == '1' else 'rackid'
                                labels.append(label)
                            label_counter = dict(Counter(labels))
                            # print(label_counter)

                            # string1=""
                            if count!=1:
                                for k, v in label_counter.items():
                                    val = "{}-{} ".format(v,k)
                                    dict1[count-1].write('found ' + val)
                            #     string1 = string1 + val

                            # st.success("Found: {}".format(string1))
                            # st.image(img, channels='BGR')
                    else:
                        if count != 1: 
                            print('file does not exist')
                            dict1[count-1].write('no_objects found')
                        # texts.append(img['text'])
                        # st.success("Found [0] Objects")
                        # st.image(img, channels='BGR')

                        
                # table creation stuff        
                df = pd.DataFrame()
                # print(texts)
                
                lst_m2 = []
                labels_m2 = []
                conf_m2 = []
                
                lst_m3 = []
                labels_m3 = []
                conf_m3 = []
                
                lst_m4 = []
                labels_m4 = []
                conf_m4 = []
                
                lst_m5 = []
                labels_m5 = []
                conf_m5 = []
                # print(runs)
                
                dw, dh, = picture.size
                
                #extracting m1 model output data - just the rack
                if os.path.exists(f"{runs[0]}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                    with open(f"{runs[0]}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                        line = f.readlines()
                        r = True
                        for dt in line:
                            label, x, y, w, h, conf = dt.split(' ')
                            if label== '1':
                                rlabel = 'Rack'
                                rl = int((float(x) - float(w) / 2) * dw)
                                rr = int((float(x) + float(w) / 2) * dw)
                                rt = int((float(y) - float(h) / 2) * dh)
                                rb = int((float(y) + float(h) / 2) * dh)
                            else:
                                r = False
                else:
                    print('No objects detected')
                    
                
                #extracting m2 model output data
                if os.path.exists(f"{runs[1]}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                    with open(f"{runs[1]}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                        line = f.readlines()
                        for dt in line:
                            label, x, y, w, h, conf = dt.split(' ')
                            conf = round(float(conf),2)
                            if label== '0':
                                label = 'Device'
                                l = int((float(x) - float(w) / 2) * dw)
                                r = int((float(x) + float(w) / 2) * dw)
                                t = int((float(y) - float(h) / 2) * dh)
                                b = int((float(y) + float(h) / 2) * dh)
                                lst_m2.append([l,r,t,b])
                                labels_m2.append(label)
                                conf_m2.append(conf)
                else:
                    print('No objects detected')
                 
                
                #extracting m3 model output data
                if os.path.exists(f"{runs[2]}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                    with open(f"{runs[2]}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                        line = f.readlines()
                        for dt in line:
                            label, x, y, w, h, conf = dt.split(' ')
                            conf = round(float(conf),2)
                            if label== '0':
                                label = 'Device'
                                l = int((float(x) - float(w) / 2) * dw)
                                r = int((float(x) + float(w) / 2) * dw)
                                t = int((float(y) - float(h) / 2) * dh)
                                b = int((float(y) + float(h) / 2) * dh)
                                lst_m3.append([l,r,t,b])
                                labels_m3.append(label)
                                conf_m3.append(conf)
                else:
                    print('No objects detected')
                
                
                #extracting m4 model output data
                if os.path.exists(f"{runs[3]}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                    with open(f"{runs[3]}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                        line = f.readlines()
                        for dt in line:
                            label, x, y, w, h, conf = dt.split(' ')
                            conf = round(float(conf),2)
                            label = 'Server' if label == '0' else 'Switch' if label == '1' else 'Router'
                            l = int((float(x) - float(w) / 2) * dw)
                            r = int((float(x) + float(w) / 2) * dw)
                            t = int((float(y) - float(h) / 2) * dh)
                            b = int((float(y) + float(h) / 2) * dh)
                            lst_m4.append([l,r,t,b])
                            labels_m4.append(label)
                            conf_m4.append(conf)
                else:
                    print('No objects detected')
                    
                
                #extracting m5 model output data
                if os.path.exists(f"{runs[4]}/labels/{uploaded_file.name.split('.')[0]}.txt"):
                    with open(f"{runs[4]}/labels/{uploaded_file.name.split('.')[0]}.txt", 'r') as f:
                        line = f.readlines()
                        for dt in line:
                            label, x, y, w, h, conf = dt.split(' ')
                            conf = round(float(conf),2)
                            label = 'Unused_Rack_Unit'
                            l = int((float(x) - float(w) / 2) * dw)
                            r = int((float(x) + float(w) / 2) * dw)
                            t = int((float(y) - float(h) / 2) * dh)
                            b = int((float(y) + float(h) / 2) * dh)
                            lst_m5.append([l,r,t,b])
                            labels_m5.append(label)
                            conf_m5.append(conf)
                else:
                    print('No objects detected')    
                 
                
                # M2 - devices 
                df['M2'] = pd.Series([f'Device_{x}' for x in np.arange(1, len(labels_m2)+1)])   
                
                #M3 - devices
                df2 = pd.DataFrame({'M3': [f'Device_{x}' for x in np.arange(1, len(labels_m3)+1)]})
                
                # print(labels_m3)
                sw, se, ro = 1, 1, 1
                ssr = []
                for k in labels_m4:
                    if k == 'Server':
                        ssr.append(k + str(se))
                        se += 1
                    elif k == 'Switch':
                        ssr.append(k +  str(sw))
                        sw += 1
                    elif k == 'Router':
                        ssr.append(k +  str(ro))
                        ro += 1
                    else: ssr.append(np.nan)
                 
                #M4 - device types
                df3 = pd.DataFrame({'M4': pd.Series(ssr)})
                
                #M5 - unused rack units
                df3_m5 = pd.DataFrame({'M5': pd.Series(labels_m5)}) 
   
                df4 = pd.concat([df,df2,df3,df3_m5], axis=1)
                
                print(texts)
                    
                df4['Bbox_M2'] = pd.Series(["".join(str(i)) for i in lst_m2])   #Bbox of M2
                df4['Bbox_M3'] = pd.Series(["".join(str(i)) for i in lst_m3])   #Bbox of M3
                df4['Bbox_M4'] = pd.Series(["".join(str(i)) for i in lst_m4])   #Bbox of M4
                df4['Bbox_M5'] = pd.Series(["".join(str(i)) for i in lst_m5])   #Bbox of M5
                
                
                # df3['Rack_name'] =  img['text']
                
#                 iou = []
#                 min_values = []
#                 m1_match_m2_new= []
#                 for count,m1 in enumerate(df3[df3['Bbox_M2'].notna()]['Bbox_M2']):
#                     m1_box = ast.literal_eval(m1)
#                     iou_lst = []
#                     min_list = []
                    
#                     for c,m2 in enumerate(df3[df3['Bbox_M3'].notna()]['Bbox_M3']):
#                         m2_box = ast.literal_eval(m2)
#                         iou_value = bb_intersection_over_union(m1_box, m2_box)
#                         min_value = distance(m1_box, m2_box)
#                         iou_lst.append(iou_value)
#                         min_list.append(min_value)

#                     val = max(iou_lst)
#                     iou.append(val)
#                     # index = iou_lst.index(val)
#                     # m1_match_m2 = df3['Bbox_M2'][index]
                    
#                     val_min = min(min_list)
#                     min_values.append(val_min)
#                     index_min = min_list.index(val_min)
#                     m1_match_m2_min = df3['Bbox_M3'][index_min]
                    
#                     # if val < 0.7:
#                     #     val = 'no match'
#                     #     index = 'no match'
#                     #     m1_match_m2 = 'no match'
                        
#                     m1_match_m2_new.append(m1_match_m2_min)
                    
                #m2 match m3
                m2_match_m3=[]
                iou_scores = []
                for count, m2 in enumerate(pd.Series(["".join(str(i)) for i in lst_m2])):
                    m2_box = ast.literal_eval(m2)
                    iou_lst =[]
                    diff = []
                    min_list = []

                    for c,m3 in enumerate(pd.Series(["".join(str(i)) for i in lst_m3])):
                        m3_box = ast.literal_eval(m3)
                        correct = distance(m2_box, m3_box)    
                        if all(i < 1500 for i in correct):
                            min_list.append(m3_box)
                            
                    if min_list:
                        for j in min_list:
                            correct_iou = bb_intersection_over_union(m2_box, j)
                            iou_lst.append(correct_iou)
                            
                        val = max(iou_lst, default=0)
                        iou_scores.append(val)
                        index = iou_lst.index(val)
                        match = min_list[index]

                        if val < 0.75:
                            val = 'no match'
                            index = 'no match'
                            match = 'no match'
                        else:
                            match = 'yes ' + str(('IOU:'+ str(round(val,2)))) + ' ' + str(min_list[index])
                    else: match = 'no match'   
                    
                    m2_match_m3.append(match)

                df4['M2_M3_match'] = pd.Series(m2_match_m3)      #M1 match in M2 model
                # df3['IOU_score[M2_M3]'] = pd.Series(iou_scores)   #IOU score between M1 and M2
                
                
                
                # m4 match in m3
                m4_match_m3=[]
                m4_match_m3_bbox = []
                iou_scores_2 = []
                for count, m4 in enumerate(pd.Series(["".join(str(i)) for i in lst_m4])):
                    m4_box = ast.literal_eval(m4)
                    iou_lst =[]
                    diff = []
                    min_list = []

                    for c, m3 in enumerate(pd.Series(["".join(str(i)) for i in lst_m3])):
                        m3_box = ast.literal_eval(m3)
                        correct = distance(m4_box, m3_box)    
                        if all(i < 1500 for i in correct):
                            min_list.append(m3_box)
                            
                    if min_list:            
                        for j in min_list:
                            correct_iou = bb_intersection_over_union(m4_box, j)
                            iou_lst.append(correct_iou)

                        val = max(iou_lst, default=0)
                        iou_scores_2.append(val)
                        index = iou_lst.index(val)
                        match = min_list[index]

                        if val < 0.75:
                            val = 'no match'
                            index = 'no match'
                            match = 'no match'
                            match1 = 'no match'
                        else:
                            match = 'yes ' + str(('IOU:' + str(round(val,2)))) + ' ' + str(min_list[index])
                            match1 = min_list[index]
                    else: 
                        match = 'no match'
                        match1 = 'no match'
                        
                    m4_match_m3.append(match)
                    m4_match_m3_bbox.append(match1)
                    
                # df4['M4_M3_match'] = pd.Series(m4_match_m3)  #M4_M3 match
                df4['M4_M3_match'] = pd.Series(m4_match_m3_bbox)
                
                # checking the count of two device models
                count1_nan = df4['Bbox_M2'].isna().sum()
                # print(count1_nan)
                count2_nan = df4['Bbox_M3'].isna().sum()
                # print(count2_nan)
                if count1_nan > count2_nan:
                    device_box = df4[df4['Bbox_M3'].notna()]['Bbox_M3']
                    device_label = df4[df4['M3'].notna()]['M3']
                    conf_m3 = pd.Series(conf_m3)
                    device_conf = conf_m3[conf_m3.notna()]
                elif count1_nan < count2_nan:
                    device_box = df4[df4['Bbox_M2'].notna()]['Bbox_M2']
                    device_label = df4[df4['M2'].notna()]['M2']
                    conf_m2 = pd.Series(conf_m2)
                    device_conf = conf_m2[conf_m2.notna()]
                elif count1_nan == count2_nan:
                    device_box = df4[df4['Bbox_M3'].notna()]['Bbox_M3']
                    device_label = df4[df4['M3'].notna()]['M3']
                    conf_m3 = pd.Series(conf_m3)
                    device_conf = conf_m3[conf_m3.notna()]
                # print(device_box)    
                
                #blank image   #Red #M2 device
                blank_image = 255*np.ones(shape=[dh,dw+1000,3],dtype=np.uint8)
                for i,label1 in zip(device_box,device_label):
                    if i != "nan":
                        i = ast.literal_eval(i)
                        cv2.rectangle(blank_image, (i[0],i[2]), (i[1], i[3]), (255, 0, 0), 20)
                        cv2.putText(blank_image, str(label1), (i[0]-500,i[2]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
                
                #blank image   #Green #M3 server switch
                for j,label2 in zip(df4[df4['Bbox_M4'].notna()]['Bbox_M4'],df4[df4['M4'].notna()]['M4']):
                    if j != "nan":
                        j = ast.literal_eval(j)
                        cv2.rectangle(blank_image, (j[0],j[2]), (j[1], j[3]), (0, 255, 0), 20)
                        cv2.putText(blank_image, str(label2), (j[0]+1100,j[2]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255, 0), 5, cv2.LINE_AA)
                
                 #blank image   #blue #M3 #unused rack
                for k,label3 in zip(df4[df4['Bbox_M5'].notna()]['Bbox_M5'],df4[df4['M5'].notna()]['M5']):
                    if k != "nan":
                        k = ast.literal_eval(k)
                        cv2.rectangle(blank_image, (k[0],k[2]), (k[1], k[3]), (0, 0, 255), 20)
                        cv2.putText(blank_image, str(label3), (k[0]+500,k[2]+150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0,255), 5, cv2.LINE_AA)
                
                
                try:
                    # blank image [rack]
                    if r or rl or rr or rt or rb:
                        cv2.rectangle(blank_image, (rl, rt), (rr, rb), (0, 0, 0), 20)
                        cv2.putText(blank_image, str(rlabel), (rl+100,rt+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0,0), 5, cv2.LINE_AA)
                

                    cv2.line(blank_image, (rl+2300, rt+100), (rl+2400, rt+100), (255, 0, 0), 20)
                    cv2.putText(blank_image, 'Device', (rl+2450,rt+100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0,0), 5, cv2.LINE_AA)

                    cv2.line(blank_image, (rl+2300, rt+200), (rl+2400, rt+200), (0, 255, 0), 20)
                    cv2.putText(blank_image, 'Device_type', (rl+2450,rt+200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255,0), 5, cv2.LINE_AA)

                    cv2.line(blank_image, (rl+2300, rt+300), (rl+2400, rt+300), (0, 0, 255), 20)
                    cv2.putText(blank_image, 'Unused_rack', (rl+2450,rt+300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0,255), 5, cv2.LINE_AA)
                
                except:
                    cv2.rectangle(blank_image, (100,100), (dw-100,dh-100), (0, 0, 0), 20)
                    cv2.putText(blank_image, 'undetected rack', (70,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0,255), 5, cv2.LINE_AA)
                
                cv2.imwrite('./layout.jpg', blank_image)
                dict1[5].write('layout')
                dict1[5].image(blank_image, use_column_width=True, channels='BGR')
                    
                df4.to_csv('./model_info.csv', index=False)    #optional 
                
                # df4['Bbox_M3'] = [ast.literal_eval(x) for x in df4[df4['Bbox_M3'].notna()]['Bbox_M3']]
                # print(type(df4['M4_M3_match']),  df4['M4_M3_match'].values, df4['Bbox_M3'].values)
                # print(df4[df4['M4_M3_match'].isin([ast.literal_eval(x) for x in df4[df4['Bbox_M3'].notna()]['Bbox_M3'].values])]['M3'])
                df4 = pd.read_csv('./model_info.csv')
                print(df4)
                # df.drop('Unnamed: 0',1)
                #creation of second table (aggregated)
                conf_m5 = pd.Series(conf_m5)
                conf_m4 = pd.Series(conf_m4)
                
                df5 = pd.DataFrame({'Devices': device_label, 
                                    'Bbox': device_box, 
                                    'Confidence': device_conf})
                
                temp = pd.DataFrame({'Devices': df4[df4['M5'].notna()]['M5'], 
                                     'Bbox': df4[df4['Bbox_M5'].notna()]['Bbox_M5'],
                                     'Confidence': conf_m5[conf_m5.notna()]})
                
                # temp1 = pd.DataFrame({'Devices': df4[df4['M4'].notna()]['M4'], 
                #                      'Bbox': df4[df4['Bbox_M4'].notna()]['Bbox_M4'],
                #                      'Confidence': conf_m4[conf_m4.notna()]})
                
                
                # # step1 - create df removing NAN and putting only those server values 
                first_df = df4.iloc[:,[2,9]]
                first_df.dropna(inplace=True)
                # print('first_df:\n',first_df)
                # # step2 - get corresponding devices
                idx = []
                for i in first_df['M4_M3_match']:
                    # print(i)
                    if i in df4['Bbox_M3'].values:
                        print("yes")
                        idx.append(df4['Bbox_M3'].loc[df4['Bbox_M3']== i].index.values.astype("int")[0])
                # print('index:',idx)
                
                findf = [df4.iloc[i,1] for i in idx]
                # print('findf:\n',findf)
                
                mid_df = pd.DataFrame(zip(first_df["M4"],first_df['M4_M3_match'],idx,findf),columns=['Device_type',"m4_m3_match","idx","findf"])
                final_df = mid_df[~mid_df['findf'].duplicated()]
                # print('final_df:\n',final_df)
                df5 = pd.concat([df5, temp]).reset_index(drop=True)
  
            
                # separating the bounding box co-ordinates
                # print(df5)
                df5['Bbox'] = [ast.literal_eval(x) for x in df5['Bbox']]
                df5[['left', 'right', 'top', 'bottom']] = pd.DataFrame(df5['Bbox'].to_list(), index=df5.index)
                df5 = df5.drop('Bbox',1)
                
                df5['device_type'] = np.NaN
                # print(df5)
                
                for i,j in zip(final_df['findf'],final_df['Device_type']):
                    # print(i)
                    # print(j)
                    df5['device_type'].loc[df5["Devices"]==i] = j
                
                df5['confidence_dtype'] = np.NaN
            
                for i,j in zip(final_df['findf'],conf_m4):
                    # print(i)
                    # print(j)
                    df5['confidence_dtype'].loc[df5["Devices"]==i] = j
                # print(df5)
                # outputs from SSR model
                # df5['Device Type'] = df4['M4']
                # df5['Confidencee'] = pd.Series(conf_m4)
                
                # inserting Rack_Name column with extracted text from tesseract
                # if texts[1]==' ' or len(texts[1])>8 or texts[1]=='':
                #     df5.insert(0, 'Rack_name', re.sub('[^A-Za-z0-9-]+', '', texts[0]))
                # else:
                #     df5.insert(0, 'Rack_name', re.sub('[^A-Za-z0-9-]+', '', texts[1]))
                #sneha
                rack_namee = uploaded_file.name
                pattern = "[A-Z]{2}\d{2,3}[A-Z]?\s*(\(\d\))?(-\d)?.(JPG|jpg)|[A-Z]{1,2}\d{2,3}.(JPG|jpg)"
                v = False
                v = re.search(pattern,rack_namee)
                if v:
                    v = v.group(0)
                    pattern1 = "[A-Z]{2}\d{2,3}[A-Z]?\s*(\(\d\))?(-\d)?|[A-Z]{1,2}\d{2,3}"
                    v1 = False
                    v1 = re.search(pattern1,v)
                    if v1:
                        df5.insert(0,'Rack_name',v1.group(0))
                    else:
                        df5.insert(0,'Rack_name',v.group(0))
                    
                else:
                    df5.insert(0,'Rack_name','no rack name')
                    
                
                
                # sorting the dataframe by top co-ordinate values
                df5 = df5.sort_values(by=['top'], ascending=True)

            # displaying first table    
            st.dataframe(df5.astype(str))
            st.write('Aggregated data')
            # displaying the second(aggregated table)
            st.dataframe(df4.astype(str))
            
            # button to download the aggregated table
            csv_file = convert_df(df4)
            if st.download_button('Press to download the above table as csv', csv_file, 'model_info.csv', 'text/csv', key='download-csv'):
                    st.write('Thanks for downloading')
                    
            # # tensorboard
            # logdir = '/home/sharedfolder/train/'
            # st_tensorboard(logdir=logdir, port=6007, width=1080)
        
if __name__ == '__main__':
    main()
