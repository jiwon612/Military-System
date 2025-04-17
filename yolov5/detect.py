# YOLOv5 
�
� 
by Ultralytics, GPL-3.0 license
 """
 Run inference on images, videos, directories, streams, etc.
 Usage- sources:
 $ python path/to/detect.py--weights yolov5s.pt--source 0
 # webcam
 img.jpg
#image
 vid.mp4
 #video
 path/
 #directory
 path/*.jpg
 #glob
 'https://youtu.be/Zgi9g1ksQHc' #YouTube
 'rtsp://example.com/media.mp4' #RTSP,RTMP,HTTPstream
 Usage-formats:
 $pythonpath/to/detect.py--weightsyolov5s.pt #
 PyTorch
 yolov5s.torchscript #
 TorchScript
 yolov5s.onnx #ONNX
 RuntimeorOpenCVDNNwith--dnn
 yolov5s.xml #
 OpenVINO
 yolov5s.engine #
 TensorRT
 yolov5s.mlmodel #
 CoreML(macOS-only)
 yolov5s_saved_model #
 TensorFlowSavedModel
 yolov5s.pb #
 TensorFlowGraphDef
 yolov5s.tflite #
 TensorFlowLite
 yolov5s_edgetpu.tflite #
 TensorFlowEdgeTPU
 """
 importargparse
 importos
 importsys
 frompathlibimportPath
 importtorch
 importtorch.backends.cudnnascudnn
 FILE=Path(__file__).resolve()
 ROOT=FILE.parents[0] #YOLOv5rootdirectory
 ifstr(ROOT)notinsys.path:
 sys.path.append(str(ROOT)) #addROOTtoPATH
 ROOT=Path(os.path.relpath(ROOT,Path.cwd())) #relative
 frommodels.commonimportDetectMultiBackend
 fromutils.dataloadersimportIMG_FORMATS,VID_FORMATS,LoadImages,
 LoadStreams
 fromutils.generalimport(LOGGER,check_file,check_img_size,
 check_imshow,check_requirements,colorstr,cv2,
 increment_path,non_max_suppression,print_args,
 scale_coords,strip_optimizer,xyxy2xywh)
 fromutils.plotsimportAnnotator,colors,save_one_box
 fromutils.torch_utilsimportselect_device,time_sync
@torch.no_grad()
 defrun(
 weights=ROOT/'yolov5s.pt', #model.ptpath(s)
 source=ROOT/'data/images', #file/dir/URL/glob,0forwebcam
 data=ROOT/'data/coco128.yaml', #dataset.yamlpath
 imgsz=(640,640), #inferencesize(height,width)
 conf_thres=0.25, #confidencethreshold
 iou_thres=0.45, #NMSIOUthreshold
 max_det=1000, #maximumdetectionsperimage
 device='', #cudadevice,i.e.0or0,1,2,3orcpu
 view_img=False, #showresults
 save_txt=False, #saveresultsto*.txt
 save_conf=False, #saveconfidencesin--save-txtlabels
 save_crop=False, #savecroppedpredictionboxes
 nosave=False, #donotsaveimages/videos
 classes=None, #filterbyclass:--class0,or--class023
 agnostic_nms=False, #class-agnosticNMS
 augment=False, #augmentedinference
 visualize=False, #visualizefeatures
 update=False, #updateallmodels
 project=ROOT/'runs/detect', #saveresultstoproject/name
 name='exp', #saveresultstoproject/name
 exist_ok=False, #existingproject/nameok,donotincrement
 line_thickness=3, #boundingboxthickness(pixels)
 hide_labels=False, #hidelabels
 hide_conf=False, #hideconfidences
 half=False, #useFP16half-precisioninference
 dnn=False, #useOpenCVDNNforONNXinference
 ):
 source=str(source)
 save_img=notnosaveandnotsource.endswith('.txt') #saveinference
 images
 is_file=Path(source).suffix[1:]in(IMG_FORMATS+VID_FORMATS)
 is_url=source.lower().startswith(('rtsp://','rtmp://','http://',
 'https://'))
 webcam=source.isnumeric()orsource.endswith('.txt')or(is_urland
 notis_file)
 ifis_urlandis_file:
 source=check_file(source) #download
 result=[]
 #Directories
 save_dir=increment_path(Path(project)/name,exist_ok=exist_ok) #
 incrementrun
 (save_dir/'labels'ifsave_txtelsesave_dir).mkdir(parents=True,
 exist_ok=True) #makedir
 #Loadmodel
 device=select_device(device)
 model=DetectMultiBackend(weights,device=device,dnn=dnn,data=data,
 fp16=half)
 stride,names,pt=model.stride,model.names,model.pt
 imgsz=check_img_size(imgsz,s=stride) #checkimagesize
 #Dataloader
 ifwebcam:
 view_img=check_imshow()
 cudnn.benchmark=True #setTruetospeedupconstantimagesize
 inference
 dataset=LoadStreams(source,img_size=imgsz,stride=stride,
auto=pt)
 bs=len(dataset) #batch_size
 else:
 dataset=LoadImages(source,img_size=imgsz,stride=stride,auto=pt)
 bs=1 #batch_size
 vid_path,vid_writer=[None]*bs,[None]*bs
 #Runinference
 model.warmup(imgsz=(1ifptelsebs,3,*imgsz)) #warmup
 seen,windows,dt=0,[],[0.0,0.0,0.0]
 forpath,im,im0s,vid_cap,sindataset:
 t1=time_sync()
 im=torch.from_numpy(im).to(device)
 im=im.half()ifmodel.fp16elseim.float() #uint8tofp16/32
 im/=255 #0-255to0.0-1.0
 iflen(im.shape)==3:
 im=im[None] #expandforbatchdim
 t2=time_sync()
 dt[0]+=t2-t1
 #Inference
 visualize=increment_path(save_dir/Path(path).stem,mkdir=True)
 ifvisualizeelseFalse
 pred=model(im,augment=augment,visualize=visualize)
 t3=time_sync()
 dt[1]+=t3-t2
 #NMS
 pred=non_max_suppression(pred,conf_thres,iou_thres,classes,
 agnostic_nms,max_det=max_det)
 dt[2]+=time_sync()-t3
 #Second-stageclassifier(optional)
 #pred=utils.general.apply_classifier(pred,classifier_model,im,
 im0s)
 #Processpredictions
 fori,detinenumerate(pred): #perimage
 seen+=1
 ifwebcam: #batch_size>=1
 p,im0,frame=path[i],im0s[i].copy(),dataset.count
 s+=f'{i}:'
 else:
 p,im0,frame=path,im0s.copy(),getattr(dataset,'frame',
 0)
 p=Path(p) #toPath
 save_path=str(save_dir/p.name) #im.jpg
 txt_path=str(save_dir/'labels'/p.stem)+(''if
 dataset.mode=='image'elsef'_{frame}') #im.txt
 s+='%gx%g'%im.shape[2:] #printstring
 gn=torch.tensor(im0.shape)[[1,0,1,0]] #normalizationgain
 whwh
 imc=im0.copy()ifsave_cropelseim0 #forsave_crop
 annotator=Annotator(im0,line_width=line_thickness,
 example=str(names))
 iflen(det):
 #Rescaleboxesfromimg_sizetoim0size
 det[:,:4]=scale_coords(im.shape[2:],det[:,:4],
 im0.shape).round()
#Printresults
 forcindet[:,-1].unique():
 n=(det[:,-1]==c).sum() #detectionsperclass
 s+=f"{n}{names[int(c)]}{'s'*(n>1)}," #addto
 string
 #Writeresults
 for*xyxy,conf,clsinreversed(det):
 #좌표출력및중심좌표계산
x1,y1,x2,y2=map(int,xyxy) #좌상단(x1,y1),
우하단(x2,y2)좌표
cx=(x1+x2)/2 #중심x좌표
cy=(y1+y2)/2 #중심y좌표
label=f"Class:{names[int(cls)]},Confidence:
 {conf:.2f},Coordinates:({x1},{y1}),({x2},{y2}),Center:({cx:.1f},
 {cy:.1f})"
 print(label)
 ifsave_txt: #Writetofile
 xywh=(xyxy2xywh(torch.tensor(xyxy).view(1,4))/
 gn).view(-1).tolist() #normalizedxywh
 line=(cls,cx,cy,conf)ifsave_confelse(cls,
 cx,cy) #labelformat
 withopen(f'{txt_path}.txt','a')asf:
 f.write(('%g'*len(line)).rstrip()%line+
 '\n')
 ifsave_imgorsave_croporview_img: #Addbboxto
 image
 c=int(cls) #integerclass
 label=Noneifhide_labelselse(names[c]if
 hide_confelsef'{names[c]}{conf:.2f}')
 annotator.box_label(xyxy,label,color=colors(c,
 True))
 ifsave_crop:
 save_one_box(xyxy,imc,file=save_dir/'crops'/
 names[c]/f'{p.stem}.jpg',BGR=True)
 #Streamresults
 im0=annotator.result()
 ifview_img:
 ifpnotinwindows:
 windows.append(p)
 cv2.namedWindow(str(p),cv2.WINDOW_NORMAL|
 cv2.WINDOW_KEEPRATIO) #allowwindowresize(Linux)
 cv2.resizeWindow(str(p),im0.shape[1],im0.shape[0])
 cv2.imshow(str(p),im0)
 cv2.waitKey(1) #1millisecond
 #Saveresults(imagewithdetections)
 ifsave_img:
 ifdataset.mode=='image':
 cv2.imwrite(save_path,im0)
 else: #'video'or'stream'
 ifvid_path[i]!=save_path: #newvideo
 vid_path[i]=save_path
 ifisinstance(vid_writer[i],cv2.VideoWriter):
 vid_writer[i].release() #releaseprevious
 videowriter
ifvid_cap: #video
 fps=vid_cap.get(cv2.CAP_PROP_FPS)
 w=int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 h=int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 else: #stream
 fps,w,h=30,im0.shape[1],im0.shape[0]
 save_path=str(Path(save_path).with_suffix('.mp4'))
 #force*.mp4suffixonresultsvideos
 vid_writer[i]=cv2.VideoWriter(save_path,
 cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
 vid_writer[i].write(im0)
 #Printtime(inference-only)
 LOGGER.info(f'{s}Done.({t3-t2:.3f}s)')
 #Printcollectedresults
 print(result)
 #Printresults
 t=tuple(x/seen*1E3forxindt) #speedsperimage
 LOGGER.info(f'Speed:%.1fmspre-process,%.1fmsinference,%.1fmsNMS
 perimageatshape{(1,3,*imgsz)}'%t)
 ifsave_txtorsave_img:
 s=f"\n{len(list(save_dir.glob('labels/*.txt')))}labelssavedto
 {save_dir/'labels'}"ifsave_txtelse''
 LOGGER.info(f"Resultssavedto{colorstr('bold',save_dir)}{s}")
 ifupdate:
 strip_optimizer(weights) #updatemodel(tofix
 SourceChangeWarning)
 defparse_opt():
 parser=argparse.ArgumentParser()
 parser.add_argument('--weights',nargs='+',type=str,default=ROOT/
 'yolov5s.pt',help='modelpath(s)')
 parser.add_argument('--source',type=str,default=ROOT/'data/images',
 help='file/dir/URL/glob,0forwebcam')
 parser.add_argument('--data',type=str,default=ROOT/
 'data/coco128.yaml',help='(optional)dataset.yamlpath')
 parser.add_argument('--imgsz','--img','--img-size',nargs='+',
 type=int,default=[640],help='inferencesizeh,w')
 parser.add_argument('--conf-thres',type=float,default=0.25,
 help='confidencethreshold')
 parser.add_argument('--iou-thres',type=float,default=0.45,help='NMS
 IoUthreshold')
 parser.add_argument('--max-det',type=int,default=1000,help='maximum
 detectionsperimage')
 parser.add_argument('--device',default='',help='cudadevice,i.e.0or
 0,1,2,3orcpu')
 parser.add_argument('--view-img',action='store_true',help='show
 results')
 parser.add_argument('--save-txt',action='store_true',help='save
 resultsto*.txt')
 parser.add_argument('--save-conf',action='store_true',help='save
 confidencesin--save-txtlabels')
 parser.add_argument('--save-crop',action='store_true',help='save
 croppedpredictionboxes')
 parser.add_argument('--nosave',action='store_true',help='donotsave
 images/videos')
 parser.add_argument('--classes',nargs='+',type=int,help='filterby
 class:--classes0,or--classes023')
 parser.add_argument('--agnostic-nms',action='store_true',
help='class-agnosticNMS')
 parser.add_argument('--augment',action='store_true',help='augmented
 inference')
 parser.add_argument('--visualize',action='store_true',help='visualize
 features')
 parser.add_argument('--update',action='store_true',help='updateall
 models')
 parser.add_argument('--project',default=ROOT/'runs/detect',
 help='saveresultstoproject/name')
 parser.add_argument('--name',default='exp',help='saveresultsto
 project/name')
 parser.add_argument('--exist-ok',action='store_true',help='existing
 project/nameok,donotincrement')
 parser.add_argument('--line-thickness',default=3,type=int,
 help='boundingboxthickness(pixels)')
 parser.add_argument('--hide-labels',default=False,action='store_true',
 help='hidelabels')
 parser.add_argument('--hide-conf',default=False,action='store_true',
 help='hideconfidences')
 parser.add_argument('--half',action='store_true',help='useFP16
 half-precisioninference')
 parser.add_argument('--dnn',action='store_true',help='useOpenCVDNN
 forONNXinference')
 opt=parser.parse_args()
 opt.imgsz*=2iflen(opt.imgsz)==1else1 #expand
 print_args(vars(opt))
 returnopt
 defmain(opt):
 check_requirements(exclude=('tensorboard','thop'))
 run(**vars(opt))
 if__name__=="__main__":
 opt=parse_opt()
 main(opt)
