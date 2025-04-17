#YOLOv5
 🚀
 byUltralytics,GPL-3.0license
 """
 Dataloadersanddatasetutils
 """
 importglob
 importhashlib
 importjson
 importmath
 importos
 importrandom
 importshutil
 importtime
 fromitertoolsimportrepeat
 frommultiprocessing.poolimportPool,ThreadPool
 frompathlibimportPath
 fromthreadingimportThread
fromurllib.parseimporturlparse
 fromzipfileimportZipFile
 importnumpyasnp
 importtorch
 importtorch.nn.functionalasF
 importyaml
 fromPILimportExifTags,Image,ImageOps
 fromtorch.utils.dataimportDataLoader,Dataset,dataloader,distributed
 fromtqdmimporttqdm
 fromutils.augmentationsimportAlbumentations,augment_hsv,copy_paste,
 letterbox,mixup,random_perspective
 fromutils.generalimport(DATASETS_DIR,LOGGER,NUM_THREADS,
 check_dataset,check_requirements,check_yaml,clean_str,
 cv2,is_colab,is_kaggle,segments2boxes,xyn2xy,
 xywh2xyxy,xywhn2xyxy,xyxy2xywhn)
 fromutils.torch_utilsimporttorch_distributed_zero_first
 #Parameters
 HELP_URL='https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
 IMG_FORMATS='bmp','dng','jpeg','jpg','mpo','png','tif','tiff',
 'webp' #includeimagesuffixes
 VID_FORMATS='asf','avi','gif','m4v','mkv','mov','mp4','mpeg',
 'mpg','ts','wmv' #includevideosuffixes
 BAR_FORMAT='{l_bar}{bar:10}{r_bar}{bar:-10b}' #tqdmbarformat
 LOCAL_RANK=int(os.getenv('LOCAL_RANK',-1)) #
 https://pytorch.org/docs/stable/elastic/run.html
 #Getorientationexiftag
 fororientationinExifTags.TAGS.keys():
 ifExifTags.TAGS[orientation]=='Orientation':
 break
 defget_hash(paths):
 #Returnsasinglehashvalueofalistofpaths(filesordirs)
 size=sum(os.path.getsize(p)forpinpathsifos.path.exists(p)) #
 sizes
 h=hashlib.md5(str(size).encode()) #hashsizes
 h.update(''.join(paths).encode()) #hashpaths
 returnh.hexdigest() #returnhash
 defexif_size(img):
 #Returnsexif-correctedPILsize
 s=img.size #(width,height)
 try:
 rotation=dict(img._getexif().items())[orientation]
 ifrotationin[6,8]: #rotation270or90
 s=(s[1],s[0])
 exceptException:
 pass
 returns
 defexif_transpose(image):
 """
 TransposeaPILimageaccordinglyifithasanEXIFOrientationtag.
Inplaceversionof
 https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py
 exif_transpose()
 :paramimage:Theimagetotranspose.
 :return:Animage.
 """
 exif=image.getexif()
 orientation=exif.get(0x0112,1) #default1
 iforientation>1:
 method={
 2:Image.FLIP_LEFT_RIGHT,
 3:Image.ROTATE_180,
 4:Image.FLIP_TOP_BOTTOM,
 5:Image.TRANSPOSE,
 6:Image.ROTATE_270,
 7:Image.TRANSVERSE,
 8:Image.ROTATE_90,}.get(orientation)
 ifmethodisnotNone:
 image=image.transpose(method)
 delexif[0x0112]
 image.info["exif"]=exif.tobytes()
 returnimage
 defcreate_dataloader(path,
 imgsz,
 batch_size,
 stride,
 single_cls=False,
 hyp=None,
 augment=False,
 cache=False,
 pad=0.0,
 rect=False,
 rank=-1,
 workers=8,
 image_weights=False,
 quad=False,
 prefix='',
 shuffle=False):
 ifrectandshuffle:
 LOGGER.warning('WARNING:--rectisincompatiblewithDataLoader
 shuffle,settingshuffle=False')
 shuffle=False
 withtorch_distributed_zero_first(rank): #initdataset*.cacheonly
 onceifDDP
 dataset=LoadImagesAndLabels(
 path,
 imgsz,
 batch_size,
 augment=augment, #augmentation
 hyp=hyp, #hyperparameters
 rect=rect, #rectangularbatches
 cache_images=cache,
 single_cls=single_cls,
 stride=int(stride),
 pad=pad,
 image_weights=image_weights,
 prefix=prefix)
batch_size=min(batch_size,len(dataset))
 nd=torch.cuda.device_count() #numberofCUDAdevices
 nw=min([os.cpu_count()//max(nd,1),batch_sizeifbatch_size>1
 else0,workers]) #numberofworkers
 sampler=Noneifrank==-1else
 distributed.DistributedSampler(dataset,shuffle=shuffle)
 loader=DataLoaderifimage_weightselseInfiniteDataLoader #only
 DataLoaderallowsforattributeupdates
 returnloader(dataset,
 batch_size=batch_size,
 shuffle=shuffleandsamplerisNone,
 num_workers=nw,
 sampler=sampler,
 pin_memory=True,
 collate_fn=LoadImagesAndLabels.collate_fn4ifquadelse
 LoadImagesAndLabels.collate_fn),dataset
 classInfiniteDataLoader(dataloader.DataLoader):
 """Dataloaderthatreusesworkers
 UsessamesyntaxasvanillaDataLoader
 """
 def__init__(self,*args,**kwargs):
 super().__init__(*args,**kwargs)
 object.__setattr__(self,'batch_sampler',
 _RepeatSampler(self.batch_sampler))
 self.iterator=super().__iter__()
 def__len__(self):
 returnlen(self.batch_sampler.sampler)
 def__iter__(self):
 for_inrange(len(self)):
 yieldnext(self.iterator)
 class_RepeatSampler:
 """Samplerthatrepeatsforever
 Args:
 sampler(Sampler)
 """
 def__init__(self,sampler):
 self.sampler=sampler
 def__iter__(self):
 whileTrue:
 yieldfromiter(self.sampler)
 classLoadImages:
 #YOLOv5image/videodataloader,i.e.`pythondetect.py--source
 image.jpg/vid.mp4`
 def__init__(self,path,img_size=640,stride=32,auto=True):
 files=[]
 forpinsorted(path)ifisinstance(path,(list,tuple))else
[path]:
 p=str(Path(p).resolve())
 if'*'inp:
 files.extend(sorted(glob.glob(p,recursive=True))) #glob
 elifos.path.isdir(p):
 files.extend(sorted(glob.glob(os.path.join(p,'*.*')))) #
 dir
 elifos.path.isfile(p):
 files.append(p) #files
 else:
 raiseFileNotFoundError(f'{p}doesnotexist')
 images=[xforxinfilesifx.split('.')[-1].lower()in
 IMG_FORMATS]
 videos=[xforxinfilesifx.split('.')[-1].lower()in
 VID_FORMATS]
 ni,nv=len(images),len(videos)
 self.img_size=img_size
 self.stride=stride
 self.files=images+videos
 self.nf=ni+nv #numberoffiles
 self.video_flag=[False]*ni+[True]*nv
 self.mode='image'
 self.auto=auto
 ifany(videos):
 self.new_video(videos[0]) #newvideo
 else:
 self.cap=None
 assertself.nf>0,f'Noimagesorvideosfoundin{p}.'\
 f'Supportedformatsare:\nimages:
 {IMG_FORMATS}\nvideos:{VID_FORMATS}'
 def__iter__(self):
 self.count=0
 returnself
 def__next__(self):
 ifself.count==self.nf:
 raiseStopIteration
 path=self.files[self.count]
 ifself.video_flag[self.count]:
 #Readvideo
 self.mode='video'
 ret_val,img0=self.cap.read()
 whilenotret_val:
 self.count+=1
 self.cap.release()
 ifself.count==self.nf: #lastvideo
 raiseStopIteration
 path=self.files[self.count]
 self.new_video(path)
 ret_val,img0=self.cap.read()
 self.frame+=1
 s=f'video{self.count+1}/{self.nf}
 ({self.frame}/{self.frames}){path}:'
 else:
#Readimage
 self.count+=1
 img0=cv2.imread(path) #BGR
 assertimg0isnotNone,f'ImageNotFound{path}'
 s=f'image{self.count}/{self.nf}{path}:'
 #Paddedresize
 img=letterbox(img0,self.img_size,stride=self.stride,
 auto=self.auto)[0]
 #Convert
 img=img.transpose((2,0,1))[::-1] #HWCtoCHW,BGRtoRGB
 img=np.ascontiguousarray(img)
 returnpath,img,img0,self.cap,s
 defnew_video(self,path):
 self.frame=0
 self.cap=cv2.VideoCapture(path)
 self.frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
 def__len__(self):
 returnself.nf #numberoffiles
 classLoadWebcam: #forinference
 #YOLOv5localwebcamdataloader,i.e.`pythondetect.py--source0`
 def__init__(self,pipe='0',img_size=640,stride=32):
 self.img_size=img_size
 self.stride=stride
 self.pipe=eval(pipe)ifpipe.isnumeric()elsepipe
 self.cap=cv2.VideoCapture(self.pipe) #videocaptureobject
 self.cap.set(cv2.CAP_PROP_BUFFERSIZE,3) #setbuffersize
 def__iter__(self):
 self.count=-1
 returnself
 def__next__(self):
 self.count+=1
 ifcv2.waitKey(1)==ord('q'): #qtoquit
 self.cap.release()
 cv2.destroyAllWindows()
 raiseStopIteration
 #Readframe
 ret_val,img0=self.cap.read()
 img0=cv2.flip(img0,1) #flipleft-right
 #Print
 assertret_val,f'CameraError{self.pipe}'
 img_path='webcam.jpg'
 s=f'webcam{self.count}:'
 #Paddedresize
 img=letterbox(img0,self.img_size,stride=self.stride)[0]
 #Convert
 img=img.transpose((2,0,1))[::-1] #HWCtoCHW,BGRtoRGB
 img=np.ascontiguousarray(img)
returnimg_path,img,img0,None,s
 def__len__(self):
 return0
 classLoadStreams:
 #YOLOv5streamloader,i.e.`pythondetect.py--source
 'rtsp://example.com/media.mp4' #RTSP,RTMP,HTTPstreams`
 def__init__(self,sources='streams.txt',img_size=640,stride=32,
 auto=True):
 self.mode='stream'
 self.img_size=img_size
 self.stride=stride
 ifos.path.isfile(sources):
 withopen(sources)asf:
 sources=[x.strip()forxinf.read().strip().splitlines()
 iflen(x.strip())]
 else:
 sources=[sources]
 n=len(sources)
 self.imgs,self.fps,self.frames,self.threads=[None]*n,[0]*
 n,[0]*n,[None]*n
 self.sources=[clean_str(x)forxinsources] #cleansourcenames
 forlater
 self.auto=auto
 fori,sinenumerate(sources): #index,source
 #Startthreadtoreadframesfromvideostream
 st=f'{i+1}/{n}:{s}...'
 ifurlparse(s).hostnamein('www.youtube.com','youtube.com',
 'youtu.be'): #ifsourceisYouTubevideo
 check_requirements(('pafy','youtube_dl==2020.12.2'))
 importpafy
 s=pafy.new(s).getbest(preftype="mp4").url #YouTubeURL
 s=eval(s)ifs.isnumeric()elses #i.e.s='0'localwebcam
 ifs==0:
 assertnotis_colab(),'--source0webcamunsupportedon
 Colab.Reruncommandinalocalenvironment.'
 assertnotis_kaggle(),'--source0webcamunsupportedon
 Kaggle.Reruncommandinalocalenvironment.'
 cap=cv2.VideoCapture(s)
 assertcap.isOpened(),f'{st}Failedtoopen{s}'
 w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 fps=cap.get(cv2.CAP_PROP_FPS) #warning:mayreturn0ornan
 self.frames[i]=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),0)
 orfloat('inf') #infinitestreamfallback
 self.fps[i]=max((fpsifmath.isfinite(fps)else0)%100,0)
 or30 #30FPSfallback
 _,self.imgs[i]=cap.read() #guaranteefirstframe
 self.threads[i]=Thread(target=self.update,args=([i,cap,s]),
 daemon=True)
 LOGGER.info(f"{st}Success({self.frames[i]}frames{w}x{h}at
 {self.fps[i]:.2f}FPS)")
 self.threads[i].start()
 LOGGER.info('') #newline
#checkforcommonshapes
 s=np.stack([letterbox(x,self.img_size,stride=self.stride,
 auto=self.auto)[0].shapeforxinself.imgs])
 self.rect=np.unique(s,axis=0).shape[0]==1 #rectinferenceif
 allshapesequal
 ifnotself.rect:
 LOGGER.warning('WARNING:Streamshapesdiffer.Foroptimal
 performancesupplysimilarly-shapedstreams.')
 defupdate(self,i,cap,stream):
 #Readstream`i`framesindaemonthread
 n,f,read=0,self.frames[i],1 #framenumber,framearray,
 inferenceevery'read'frame
 whilecap.isOpened()andn<f:
 n+=1
 #_,self.imgs[index]=cap.read()
 cap.grab()
 ifn%read==0:
 success,im=cap.retrieve()
 ifsuccess:
 self.imgs[i]=im
 else:
 LOGGER.warning('WARNING:Videostreamunresponsive,
 pleasecheckyourIPcameraconnection.')
 self.imgs[i]=np.zeros_like(self.imgs[i])
 cap.open(stream) #re-openstreamifsignalwaslost
 time.sleep(0.0) #waittime
 def__iter__(self):
 self.count=-1
 returnself
 def__next__(self):
 self.count+=1
 ifnotall(x.is_alive()forxinself.threads)orcv2.waitKey(1)==
 ord('q'): #qtoquit
 cv2.destroyAllWindows()
 raiseStopIteration
 #Letterbox
 img0=self.imgs.copy()
 img=[letterbox(x,self.img_size,stride=self.stride,
 auto=self.rectandself.auto)[0]forxinimg0]
 #Stack
 img=np.stack(img,0)
 #Convert
 img=img[...,::-1].transpose((0,3,1,2)) #BGRtoRGB,BHWCto
 BCHW
 img=np.ascontiguousarray(img)
 returnself.sources,img,img0,None,''
 def__len__(self):
 returnlen(self.sources) #1E12frames=32streamsat30FPSfor
 30years
defimg2label_paths(img_paths):
 #Definelabelpathsasafunctionofimagepaths
 sa,sb=f'{os.sep}images{os.sep}',f'{os.sep}labels{os.sep}' #
 /images/,/labels/substrings
 return[sb.join(x.rsplit(sa,1)).rsplit('.',1)[0]+'.txt'forxin
 img_paths]
 classLoadImagesAndLabels(Dataset):
 #YOLOv5train_loader/val_loader,loadsimagesandlabelsfortraining
 andvalidation
 cache_version=0.6 #datasetlabels*.cacheversion
 rand_interp_methods=[cv2.INTER_NEAREST,cv2.INTER_LINEAR,
 cv2.INTER_CUBIC,cv2.INTER_AREA,cv2.INTER_LANCZOS4]
 def__init__(self,
 path,
 img_size=640,
 batch_size=16,
 augment=False,
 hyp=None,
 rect=False,
 image_weights=False,
 cache_images=False,
 single_cls=False,
 stride=32,
 pad=0.0,
 prefix=''):
 self.img_size=img_size
 self.augment=augment
 self.hyp=hyp
 self.image_weights=image_weights
 self.rect=Falseifimage_weightselserect
 self.mosaic=self.augmentandnotself.rect #load4imagesata
 timeintoamosaic(onlyduringtraining)
 self.mosaic_border=[-img_size//2,-img_size//2]
 self.stride=stride
 self.path=path
 self.albumentations=Albumentations()ifaugmentelseNone
 try:
 f=[] #imagefiles
 forpinpathifisinstance(path,list)else[path]:
 p=Path(p) #os-agnostic
 ifp.is_dir(): #dir
 f+=glob.glob(str(p/'**'/'*.*'),recursive=True)
 #f=list(p.rglob('*.*')) #pathlib
 elifp.is_file(): #file
 withopen(p)ast:
 t=t.read().strip().splitlines()
 parent=str(p.parent)+os.sep
 f+=[x.replace('./',parent)ifx.startswith('./')
 elsexforxint] #localtoglobalpath
 #f+=[p.parent/x.lstrip(os.sep)forxint] #
 localtoglobalpath(pathlib)
 else:
 raiseFileNotFoundError(f'{prefix}{p}doesnotexist')
 self.im_files=sorted(x.replace('/',os.sep)forxinfif
 x.split('.')[-1].lower()inIMG_FORMATS)
 #self.img_files=sorted([xforxinfifx.suffix[1:].lower()
inIMG_FORMATS]) #pathlib
 assertself.im_files,f'{prefix}Noimagesfound'
 exceptExceptionase:
 raiseException(f'{prefix}Errorloadingdatafrom{path}:
 {e}\nSee{HELP_URL}')
 #Checkcache
 self.label_files=img2label_paths(self.im_files) #labels
 cache_path=(pifp.is_file()else
 Path(self.label_files[0]).parent).with_suffix('.cache')
 try:
 cache,exists=np.load(cache_path,allow_pickle=True).item(),
 True #loaddict
 assertcache['version']==self.cache_version #matchescurrent
 version
 assertcache['hash']==get_hash(self.label_files+
 self.im_files) #identicalhash
 exceptException:
 cache,exists=self.cache_labels(cache_path,prefix),False #
 runcacheops
 #Displaycache
 nf,nm,ne,nc,n=cache.pop('results') #found,missing,empty,
 corrupt,total
 ifexistsandLOCAL_RANKin{-1,0}:
 d=f"Scanning'{cache_path}'imagesandlabels...{nf}found,
 {nm}missing,{ne}empty,{nc}corrupt"
 tqdm(None,desc=prefix+d,total=n,initial=n,
 bar_format=BAR_FORMAT) #displaycacheresults
 ifcache['msgs']:
 LOGGER.info('\n'.join(cache['msgs'])) #displaywarnings
 assertnf>0ornotaugment,f'{prefix}Nolabelsin{cache_path}.
 Cannottrainwithoutlabels.See{HELP_URL}'
 #Readcache
 [cache.pop(k)forkin('hash','version','msgs')] #removeitems
 labels,shapes,self.segments=zip(*cache.values())
 self.labels=list(labels)
 self.shapes=np.array(shapes,dtype=np.float64)
 self.im_files=list(cache.keys()) #update
 self.label_files=img2label_paths(cache.keys()) #update
 n=len(shapes) #numberofimages
 bi=np.floor(np.arange(n)/batch_size).astype(int) #batchindex
 nb=bi[-1]+1 #numberofbatches
 self.batch=bi #batchindexofimage
 self.n=n
 self.indices=range(n)
 #Updatelabels
 include_class=[] #filterlabelstoincludeonlytheseclasses
 (optional)
 include_class_array=np.array(include_class).reshape(1,-1)
 fori,(label,segment)inenumerate(zip(self.labels,
 self.segments)):
 ifinclude_class:
 j=(label[:,0:1]==include_class_array).any(1)
 self.labels[i]=label[j]
 ifsegment:
 self.segments[i]=segment[j]
 ifsingle_cls: #single-classtraining,mergeallclassesinto
0
 self.labels[i][:,0]=0
 ifsegment:
 self.segments[i][:,0]=0
 #RectangularTraining
 ifself.rect:
 #Sortbyaspectratio
 s=self.shapes #wh
 ar=s[:,1]/s[:,0] #aspectratio
 irect=ar.argsort()
 self.im_files=[self.im_files[i]foriinirect]
 self.label_files=[self.label_files[i]foriinirect]
 self.labels=[self.labels[i]foriinirect]
 self.shapes=s[irect] #wh
 ar=ar[irect]
 #Settrainingimageshapes
 shapes=[[1,1]]*nb
 foriinrange(nb):
 ari=ar[bi==i]
 mini,maxi=ari.min(),ari.max()
 ifmaxi<1:
 shapes[i]=[maxi,1]
 elifmini>1:
 shapes[i]=[1,1/mini]
 self.batch_shapes=np.ceil(np.array(shapes)*img_size/stride
 +pad).astype(int)*stride
 #CacheimagesintoRAM/diskforfastertraining(WARNING:large
 datasetsmayexceedsystemresources)
 self.ims=[None]*n
 self.npy_files=[Path(f).with_suffix('.npy')forfin
 self.im_files]
 ifcache_images:
 gb=0 #Gigabytesofcachedimages
 self.im_hw0,self.im_hw=[None]*n,[None]*n
 fcn=self.cache_images_to_diskifcache_images=='disk'else
 self.load_image
 results=ThreadPool(NUM_THREADS).imap(fcn,range(n))
 pbar=tqdm(enumerate(results),total=n,bar_format=BAR_FORMAT,
 disable=LOCAL_RANK>0)
 fori,xinpbar:
 ifcache_images=='disk':
 gb+=self.npy_files[i].stat().st_size
 else: #'ram'
 self.ims[i],self.im_hw0[i],self.im_hw[i]=x #im,
 hw_orig,hw_resized=load_image(self,i)
 gb+=self.ims[i].nbytes
 pbar.desc=f'{prefix}Cachingimages({gb/1E9:.1f}GB
 {cache_images})'
 pbar.close()
 defcache_labels(self,path=Path('./labels.cache'),prefix=''):
 #Cachedatasetlabels,checkimagesandreadshapes
 x={} #dict
 nm,nf,ne,nc,msgs=0,0,0,0,[] #numbermissing,found,
 empty,corrupt,messages
 desc=f"{prefix}Scanning'{path.parent/path.stem}'imagesand
labels..."
 withPool(NUM_THREADS)aspool:
 pbar=tqdm(pool.imap(verify_image_label,zip(self.im_files,
 self.label_files,repeat(prefix))),
 desc=desc,
 total=len(self.im_files),
 bar_format=BAR_FORMAT)
 forim_file,lb,shape,segments,nm_f,nf_f,ne_f,nc_f,msgin
 pbar:
 nm+=nm_f
 nf+=nf_f
 ne+=ne_f
 nc+=nc_f
 ifim_file:
 x[im_file]=[lb,shape,segments]
 ifmsg:
 msgs.append(msg)
 pbar.desc=f"{desc}{nf}found,{nm}missing,{ne}empty,
 {nc}corrupt"
 pbar.close()
 ifmsgs:
 LOGGER.info('\n'.join(msgs))
 ifnf==0:
 LOGGER.warning(f'{prefix}WARNING:Nolabelsfoundin{path}.See
 {HELP_URL}')
 x['hash']=get_hash(self.label_files+self.im_files)
 x['results']=nf,nm,ne,nc,len(self.im_files)
 x['msgs']=msgs #warnings
 x['version']=self.cache_version #cacheversion
 try:
 np.save(path,x) #savecachefornexttime
 path.with_suffix('.cache.npy').rename(path) #remove.npy
 suffix
 LOGGER.info(f'{prefix}Newcachecreated:{path}')
 exceptExceptionase:
 LOGGER.warning(f'{prefix}WARNING:Cachedirectory{path.parent}
 isnotwriteable:{e}') #notwriteable
 returnx
 def__len__(self):
 returnlen(self.im_files)
 #def__iter__(self):
 # self.count=-1
 # print('randatasetiter')
 # #self.shuffled_vector=np.random.permutation(self.nF)if
 self.augmentelsenp.arange(self.nF)
 # returnself
 def__getitem__(self,index):
 index=self.indices[index] #linear,shuffled,orimage_weights
 hyp=self.hyp
 mosaic=self.mosaicandrandom.random()<hyp['mosaic']
 ifmosaic:
 #Loadmosaic
 img,labels=self.load_mosaic(index)
 shapes=None
#MixUpaugmentation
 ifrandom.random()<hyp['mixup']:
 img,labels=mixup(img,labels,
 *self.load_mosaic(random.randint(0,self.n-1)))
 else:
 #Loadimage
 img,(h0,w0),(h,w)=self.load_image(index)
 #Letterbox
 shape=self.batch_shapes[self.batch[index]]ifself.rectelse
 self.img_size #finalletterboxedshape
 img,ratio,pad=letterbox(img,shape,auto=False,
 scaleup=self.augment)
 shapes=(h0,w0),((h/h0,w/w0),pad) #forCOCOmAP
 rescaling
 labels=self.labels[index].copy()
 iflabels.size: #normalizedxywhtopixelxyxyformat
 labels[:,1:]=xywhn2xyxy(labels[:,1:],ratio[0]*w,
 ratio[1]*h,padw=pad[0],padh=pad[1])
 ifself.augment:
 img,labels=random_perspective(img,
 labels,
 degrees=hyp['degrees'],
 translate=hyp['translate'],
 scale=hyp['scale'],
 shear=hyp['shear'],
 perspective=hyp['perspective'])
 nl=len(labels) #numberoflabels
 ifnl:
 labels[:,1:5]=xyxy2xywhn(labels[:,1:5],w=img.shape[1],
 h=img.shape[0],clip=True,eps=1E-3)
 ifself.augment:
 #Albumentations
 img,labels=self.albumentations(img,labels)
 nl=len(labels) #updateafteralbumentations
 #HSVcolor-space
 augment_hsv(img,hgain=hyp['hsv_h'],sgain=hyp['hsv_s'],
 vgain=hyp['hsv_v'])
 #Flipup-down
 ifrandom.random()<hyp['flipud']:
 img=np.flipud(img)
 ifnl:
 labels[:,2]=1-labels[:,2]
 #Flipleft-right
 ifrandom.random()<hyp['fliplr']:
 img=np.fliplr(img)
 ifnl:
 labels[:,1]=1-labels[:,1]
 #Cutouts
 #labels=cutout(img,labels,p=0.5)
#nl=len(labels) #updateaftercutout
 labels_out=torch.zeros((nl,6))
 ifnl:
 labels_out[:,1:]=torch.from_numpy(labels)
 #Convert
 img=img.transpose((2,0,1))[::-1] #HWCtoCHW,BGRtoRGB
 img=np.ascontiguousarray(img)
 returntorch.from_numpy(img),labels_out,self.im_files[index],
 shapes
 defload_image(self,i):
 #Loads1imagefromdatasetindex'i',returns(im,originalhw,
 resizedhw)
 im,f,fn=self.ims[i],self.im_files[i],self.npy_files[i],
 ifimisNone: #notcachedinRAM
 iffn.exists(): #loadnpy
 im=np.load(fn)
 else: #readimage
 im=cv2.imread(f) #BGR
 assertimisnotNone,f'ImageNotFound{f}'
 h0,w0=im.shape[:2] #orighw
 r=self.img_size/max(h0,w0) #ratio
 ifr!=1: #ifsizesarenotequal
 interp=cv2.INTER_LINEARif(self.augmentorr>1)else
 cv2.INTER_AREA
 im=cv2.resize(im,(int(w0*r),int(h0*r)),
 interpolation=interp)
 returnim,(h0,w0),im.shape[:2] #im,hw_original,hw_resized
 else:
 returnself.ims[i],self.im_hw0[i],self.im_hw[i] #im,
 hw_original,hw_resized
 defcache_images_to_disk(self,i):
 #Savesanimageasan*.npyfileforfasterloading
 f=self.npy_files[i]
 ifnotf.exists():
 np.save(f.as_posix(),cv2.imread(self.im_files[i]))
 defload_mosaic(self,index):
 #YOLOv54-mosaicloader.Loads1image+3randomimagesintoa
 4-imagemosaic
 labels4,segments4=[],[]
 s=self.img_size
 yc,xc=(int(random.uniform(-x,2*s+x))forxin
 self.mosaic_border) #mosaiccenterx,y
 indices=[index]+random.choices(self.indices,k=3) #3
 additionalimageindices
 random.shuffle(indices)
 fori,indexinenumerate(indices):
 #Loadimage
 img,_,(h,w)=self.load_image(index)
 #placeimginimg4
 ifi==0: #topleft
 img4=np.full((s*2,s*2,img.shape[2]),114,
 dtype=np.uint8) #baseimagewith4tiles
 x1a,y1a,x2a,y2a=max(xc-w,0),max(yc-h,0),xc,yc
#xmin,ymin,xmax,ymax(largeimage)
 x1b,y1b,x2b,y2b=w-(x2a-x1a),h-(y2a-y1a),w,h
 #xmin,ymin,xmax,ymax(smallimage)
 elifi==1: #topright
 x1a,y1a,x2a,y2a=xc,max(yc-h,0),min(xc+w,s*2),
 yc
 x1b,y1b,x2b,y2b=0,h-(y2a-y1a),min(w,x2a-x1a),
 h
 elifi==2: #bottomleft
 x1a,y1a,x2a,y2a=max(xc-w,0),yc,xc,min(s*2,yc+
 h)
 x1b,y1b,x2b,y2b=w-(x2a-x1a),0,w,min(y2a-y1a,
 h)
 elifi==3: #bottomright
 x1a,y1a,x2a,y2a=xc,yc,min(xc+w,s*2),min(s*2,
 yc+h)
 x1b,y1b,x2b,y2b=0,0,min(w,x2a-x1a),min(y2a-y1a,
 h)
 img4[y1a:y2a,x1a:x2a]=img[y1b:y2b,x1b:x2b] #
 img4[ymin:ymax,xmin:xmax]
 padw=x1a-x1b
 padh=y1a-y1b
 #Labels
 labels,segments=self.labels[index].copy(),
 self.segments[index].copy()
 iflabels.size:
 labels[:,1:]=xywhn2xyxy(labels[:,1:],w,h,padw,padh)
 #normalizedxywhtopixelxyxyformat
 segments=[xyn2xy(x,w,h,padw,padh)forxinsegments]
 labels4.append(labels)
 segments4.extend(segments)
 #Concat/cliplabels
 labels4=np.concatenate(labels4,0)
 forxin(labels4[:,1:],*segments4):
 np.clip(x,0,2*s,out=x) #clipwhenusing
 random_perspective()
 #img4,labels4=replicate(img4,labels4) #replicate
 #Augment
 img4,labels4,segments4=copy_paste(img4,labels4,segments4,
 p=self.hyp['copy_paste'])
 img4,labels4=random_perspective(img4,
 labels4,
 segments4,
 degrees=self.hyp['degrees'],
 translate=self.hyp['translate'],
 scale=self.hyp['scale'],
 shear=self.hyp['shear'],
 perspective=self.hyp['perspective'],
 border=self.mosaic_border) #
 bordertoremove
 returnimg4,labels4
 defload_mosaic9(self,index):
 #YOLOv59-mosaicloader.Loads1image+8randomimagesintoa
9-imagemosaic
 labels9,segments9=[],[]
 s=self.img_size
 indices=[index]+random.choices(self.indices,k=8) #8
 additionalimageindices
 random.shuffle(indices)
 hp,wp=-1,-1 #height,widthprevious
 fori,indexinenumerate(indices):
 #Loadimage
 img,_,(h,w)=self.load_image(index)
 #placeimginimg9
 ifi==0: #center
 img9=np.full((s*3,s*3,img.shape[2]),114,
 dtype=np.uint8) #baseimagewith4tiles
 h0,w0=h,w
 c=s,s,s+w,s+h #xmin,ymin,xmax,ymax(base)
 coordinates
 elifi==1: #top
 c=s,s-h,s+w,s
 elifi==2: #topright
 c=s+wp,s-h,s+wp+w,s
 elifi==3: #right
 c=s+w0,s,s+w0+w,s+h
 elifi==4: #bottomright
 c=s+w0,s+hp,s+w0+w,s+hp+h
 elifi==5: #bottom
 c=s+w0-w,s+h0,s+w0,s+h0+h
 elifi==6: #bottomleft
 c=s+w0-wp-w,s+h0,s+w0-wp,s+h0+h
 elifi==7: #left
 c=s-w,s+h0-h,s,s+h0
 elifi==8: #topleft
 c=s-w,s+h0-hp-h,s,s+h0-hp
 padx,pady=c[:2]
 x1,y1,x2,y2=(max(x,0)forxinc) #allocatecoords
 #Labels
 labels,segments=self.labels[index].copy(),
 self.segments[index].copy()
 iflabels.size:
 labels[:,1:]=xywhn2xyxy(labels[:,1:],w,h,padx,pady)
 #normalizedxywhtopixelxyxyformat
 segments=[xyn2xy(x,w,h,padx,pady)forxinsegments]
 labels9.append(labels)
 segments9.extend(segments)
 #Image
 img9[y1:y2,x1:x2]=img[y1-pady:,x1-padx:] #
 img9[ymin:ymax,xmin:xmax]
 hp,wp=h,w #height,widthprevious
 #Offset
 yc,xc=(int(random.uniform(0,s))for_inself.mosaic_border) #
 mosaiccenterx,y
 img9=img9[yc:yc+2*s,xc:xc+2*s]
 #Concat/cliplabels
 labels9=np.concatenate(labels9,0)
labels9[:,[1,3]]-=xc
 labels9[:,[2,4]]-=yc
 c=np.array([xc,yc]) #centers
 segments9=[x-cforxinsegments9]
 forxin(labels9[:,1:],*segments9):
 np.clip(x,0,2*s,out=x) #clipwhenusing
 random_perspective()
 #img9,labels9=replicate(img9,labels9) #replicate
 #Augment
 img9,labels9=random_perspective(img9,
 labels9,
 segments9,
 degrees=self.hyp['degrees'],
 translate=self.hyp['translate'],
 scale=self.hyp['scale'],
 shear=self.hyp['shear'],
 perspective=self.hyp['perspective'],
 border=self.mosaic_border) #
 bordertoremove
 returnimg9,labels9
 @staticmethod
 defcollate_fn(batch):
 im,label,path,shapes=zip(*batch) #transposed
 fori,lbinenumerate(label):
 lb[:,0]=i #addtargetimageindexforbuild_targets()
 returntorch.stack(im,0),torch.cat(label,0),path,shapes
 @staticmethod
 defcollate_fn4(batch):
 img,label,path,shapes=zip(*batch) #transposed
 n=len(shapes)//4
 im4,label4,path4,shapes4=[],[],path[:n],shapes[:n]
 ho=torch.tensor([[0.0,0,0,1,0,0]])
 wo=torch.tensor([[0.0,0,1,0,0,0]])
 s=torch.tensor([[1,1,0.5,0.5,0.5,0.5]]) #scale
 foriinrange(n): #zidanetorch.zeros(16,3,720,1280) #BCHW
 i*=4
 ifrandom.random()<0.5:
 im=F.interpolate(img[i].unsqueeze(0).float(),
 scale_factor=2.0,mode='bilinear',
 align_corners=False)[0].type(img[i].type())
 lb=label[i]
 else:
 im=torch.cat((torch.cat((img[i],img[i+1]),1),
 torch.cat((img[i+2],img[i+3]),1)),2)
 lb=torch.cat((label[i],label[i+1]+ho,label[i+2]+
 wo,label[i+3]+ho+wo),0)*s
 im4.append(im)
 label4.append(lb)
 fori,lbinenumerate(label4):
 lb[:,0]=i #addtargetimageindexforbuild_targets()
returntorch.stack(im4,0),torch.cat(label4,0),path4,shapes4
 #Ancillaryfunctions--------------------------------------------------------------------------------------------------
 defcreate_folder(path='./new'):
 #Createfolder
 ifos.path.exists(path):
 shutil.rmtree(path) #deleteoutputfolder
 os.makedirs(path) #makenewoutputfolder
 defflatten_recursive(path=DATASETS_DIR/'coco128'):
 #Flattenarecursivedirectorybybringingallfilestotoplevel
 new_path=Path(str(path)+'_flat')
 create_folder(new_path)
 forfileintqdm(glob.glob(str(Path(path))+'/**/*.*',
 recursive=True)):
 shutil.copyfile(file,new_path/Path(file).name)
 defextract_boxes(path=DATASETS_DIR/'coco128'): #fromutils.dataloaders
 import*;extract_boxes()
 #Convertdetectiondatasetintoclassificationdataset,withone
 directoryperclass
 path=Path(path) #imagesdir
 shutil.rmtree(path/'classifier')if(path/'classifier').is_dir()
 elseNone #removeexisting
 files=list(path.rglob('*.*'))
 n=len(files) #numberoffiles
 forim_fileintqdm(files,total=n):
 ifim_file.suffix[1:]inIMG_FORMATS:
 #image
 im=cv2.imread(str(im_file))[...,::-1] #BGRtoRGB
 h,w=im.shape[:2]
 #labels
 lb_file=Path(img2label_paths([str(im_file)])[0])
 ifPath(lb_file).exists():
 withopen(lb_file)asf:
 lb=np.array([x.split()forxin
 f.read().strip().splitlines()],dtype=np.float32) #labels
 forj,xinenumerate(lb):
 c=int(x[0]) #class
 f=(path/'classifier')/f'{c}'/
 f'{path.stem}_{im_file.stem}_{j}.jpg' #newfilename
 ifnotf.parent.is_dir():
 f.parent.mkdir(parents=True)
 b=x[1:]*[w,h,w,h] #box
 #b[2:]=b[2:].max() #rectangletosquare
 b[2:]=b[2:]*1.2+3 #pad
 b=xywh2xyxy(b.reshape(-1,4)).ravel().astype(int)
 b[[0,2]]=np.clip(b[[0,2]],0,w) #clipboxes
 outsideofimage
 b[[1,3]]=np.clip(b[[1,3]],0,h)
 assertcv2.imwrite(str(f),im[b[1]:b[3],b[0]:b[2]]),
f'boxfailurein{f}'
 defautosplit(path=DATASETS_DIR/'coco128/images',weights=(0.9,0.1,
 0.0),annotated_only=False):
 """Autosplitadatasetintotrain/val/testsplitsandsave
 path/autosplit_*.txtfiles
 Usage:fromutils.dataloadersimport*;autosplit()
 Arguments
 path: Pathtoimagesdirectory
 weights: Train,val,testweights(list,tuple)
 annotated_only: Onlyuseimageswithanannotatedtxtfile
 """
 path=Path(path) #imagesdir
 files=sorted(xforxinpath.rglob('*.*')ifx.suffix[1:].lower()in
 IMG_FORMATS) #imagefilesonly
 n=len(files) #numberoffiles
 random.seed(0) #forreproducibility
 indices=random.choices([0,1,2],weights=weights,k=n) #assigneach
 imagetoasplit
 txt=['autosplit_train.txt','autosplit_val.txt','autosplit_test.txt']
 #3txtfiles
 [(path.parent/x).unlink(missing_ok=True)forxintxt] #remove
 existing
 print(f'Autosplittingimagesfrom{path}'+',using*.txtlabeled
 imagesonly'*annotated_only)
 fori,imgintqdm(zip(indices,files),total=n):
 ifnotannotated_onlyor
 Path(img2label_paths([str(img)])[0]).exists(): #checklabel
 withopen(path.parent/txt[i],'a')asf:
 f.write('./'+img.relative_to(path.parent).as_posix()+
 '\n') #addimagetotxtfile
 defverify_image_label(args):
 #Verifyoneimage-labelpair
 im_file,lb_file,prefix=args
 nm,nf,ne,nc,msg,segments=0,0,0,0,'',[] #number(missing,
 found,empty,corrupt),message,segments
 try:
 #verifyimages
 im=Image.open(im_file)
 im.verify() #PILverify
 shape=exif_size(im) #imagesize
 assert(shape[0]>9)&(shape[1]>9),f'imagesize{shape}<10
 pixels'
 assertim.format.lower()inIMG_FORMATS,f'invalidimageformat
 {im.format}'
 ifim.format.lower()in('jpg','jpeg'):
 withopen(im_file,'rb')asf:
 f.seek(-2,2)
 iff.read()!=b'\xff\xd9': #corruptJPEG
 ImageOps.exif_transpose(Image.open(im_file)).save(im_file,'JPEG',
 subsampling=0,quality=100)
 msg=f'{prefix}WARNING:{im_file}:corruptJPEG
 restoredandsaved'
#verifylabels
 ifos.path.isfile(lb_file):
 nf=1 #labelfound
 withopen(lb_file)asf:
 lb=[x.split()forxinf.read().strip().splitlines()if
 len(x)]
 ifany(len(x)>6forxinlb): #issegment
 classes=np.array([x[0]forxinlb],dtype=np.float32)
 segments=[np.array(x[1:],
 dtype=np.float32).reshape(-1,2)forxinlb] #(cls,xy1...)
 lb=np.concatenate((classes.reshape(-1,1),
 segments2boxes(segments)),1) #(cls,xywh)
 lb=np.array(lb,dtype=np.float32)
 nl=len(lb)
 ifnl:
 assertlb.shape[1]==5,f'labelsrequire5columns,
 {lb.shape[1]}columnsdetected'
 assert(lb>=0).all(),f'negativelabelvalues{lb[lb<
 0]}'
 assert(lb[:,1:]<=1).all(),f'non-normalizedoroutof
 boundscoordinates{lb[:,1:][lb[:,1:]>1]}'
 _,i=np.unique(lb,axis=0,return_index=True)
 iflen(i)<nl: #duplicaterowcheck
 lb=lb[i] #removeduplicates
 ifsegments:
 segments=segments[i]
 msg=f'{prefix}WARNING:{im_file}:{nl-len(i)}
 duplicatelabelsremoved'
 else:
 ne=1 #labelempty
 lb=np.zeros((0,5),dtype=np.float32)
 else:
 nm=1 #labelmissing
 lb=np.zeros((0,5),dtype=np.float32)
 returnim_file,lb,shape,segments,nm,nf,ne,nc,msg
 exceptExceptionase:
 nc=1
 msg=f'{prefix}WARNING:{im_file}:ignoringcorruptimage/label:
 {e}'
 return[None,None,None,None,nm,nf,ne,nc,msg]
 defdataset_stats(path='coco128.yaml',autodownload=False,verbose=False,
 profile=False,hub=False):
 """Returndatasetstatisticsdictionarywithimagesandinstances
 countspersplitperclass
 Toruninparentdirectory:exportPYTHONPATH="$PWD/yolov5"
 Usage1:fromutils.dataloadersimport*;dataset_stats('coco128.yaml',
 autodownload=True)
 Usage2:fromutils.dataloadersimport*;
 dataset_stats('path/to/coco128_with_yaml.zip')
 Arguments
 path: Pathtodata.yamlordata.zip(withdata.yaml
 insidedata.zip)
 autodownload: Attempttodownloaddatasetifnotfoundlocally
 verbose: Printstatsdictionary
 """
 def_round_labels(labels):
 #Updatelabelstointegerclassand6decimalplacefloats
return[[int(c),*(round(x,4)forxinpoints)]forc,*pointsin
 labels]
 def_find_yaml(dir):
 #Returndata.yamlfile
 files=list(dir.glob('*.yaml'))orlist(dir.rglob('*.yaml')) #try
 rootlevelfirstandthenrecursive
 assertfiles,f'No*.yamlfilefoundin{dir}'
 iflen(files)>1:
 files=[fforfinfilesiff.stem==dir.stem] #prefer
 *.yamlfilesthatmatchdirname
 assertfiles,f'Multiple*.yamlfilesfoundin{dir},only1
 *.yamlfileallowed'
 assertlen(files)==1,f'Multiple*.yamlfilesfound:{files},only
 1*.yamlfileallowedin{dir}'
 returnfiles[0]
 def_unzip(path):
 #Unzipdata.zip
 ifstr(path).endswith('.zip'): #pathisdata.zip
 assertPath(path).is_file(),f'Errorunzipping{path},filenot
 found'
 ZipFile(path).extractall(path=path.parent) #unzip
 dir=path.with_suffix('') #datasetdirectory==zipname
 assertdir.is_dir(),f'Errorunzipping{path},{dir}notfound.
 path/to/abc.zipMUSTunziptopath/to/abc/'
 returnTrue,str(dir),_find_yaml(dir) #zipped,data_dir,
 yaml_path
 else: #pathisdata.yaml
 returnFalse,None,path
 def_hub_ops(f,max_dim=1920):
 #HUBopsfor1image'f':resizeandsaveatreducedqualityin
 /dataset-hubforweb/appviewing
 f_new=im_dir/Path(f).name #dataset-hubimagefilename
 try: #usePIL
 im=Image.open(f)
 r=max_dim/max(im.height,im.width) #ratio
 ifr<1.0: #imagetoolarge
 im=im.resize((int(im.width*r),int(im.height*r)))
 im.save(f_new,'JPEG',quality=75,optimize=True) #save
 exceptExceptionase: #useOpenCV
 print(f'WARNING:HUBopsPILfailure{f}:{e}')
 im=cv2.imread(f)
 im_height,im_width=im.shape[:2]
 r=max_dim/max(im_height,im_width) #ratio
 ifr<1.0: #imagetoolarge
 im=cv2.resize(im,(int(im_width*r),int(im_height*r)),
 interpolation=cv2.INTER_AREA)
 cv2.imwrite(str(f_new),im)
 zipped,data_dir,yaml_path=_unzip(Path(path))
 try:
 withopen(check_yaml(yaml_path),errors='ignore')asf:
 data=yaml.safe_load(f) #datadict
 ifzipped:
 data['path']=data_dir #TODO:shouldthisbe
 dir.resolve()?`
 exceptException:
 raiseException("error/HUB/dataset_stats/yaml_load")
check_dataset(data,autodownload) #downloaddatasetifmissing
 hub_dir=Path(data['path']+('-hub'ifhubelse''))
 stats={'nc':data['nc'],'names':data['names']} #statistics
 dictionary
 forsplitin'train','val','test':
 ifdata.get(split)isNone:
 stats[split]=None #i.e.notestset
 continue
 x=[]
 dataset=LoadImagesAndLabels(data[split]) #loaddataset
 forlabelintqdm(dataset.labels,total=dataset.n,
 desc='Statistics'):
 x.append(np.bincount(label[:,0].astype(int),
 minlength=data['nc']))
 x=np.array(x) #shape(128x80)
 stats[split]={
 'instance_stats':{
 'total':int(x.sum()),
 'per_class':x.sum(0).tolist()},
 'image_stats':{
 'total':dataset.n,
 'unlabelled':int(np.all(x==0,1).sum()),
 'per_class':(x>0).sum(0).tolist()},
 'labels':[{
 str(Path(k).name):_round_labels(v.tolist())}fork,vin
 zip(dataset.im_files,dataset.labels)]}
 ifhub:
 im_dir=hub_dir/'images'
 im_dir.mkdir(parents=True,exist_ok=True)
 for_intqdm(ThreadPool(NUM_THREADS).imap(_hub_ops,
 dataset.im_files),total=dataset.n,desc='HUBOps'):
 pass
 #Profile
 stats_path=hub_dir/'stats.json'
 ifprofile:
 for_inrange(1):
 file=stats_path.with_suffix('.npy')
 t1=time.time()
 np.save(file,stats)
 t2=time.time()
 x=np.load(file,allow_pickle=True)
 print(f'stats.npytimes:{time.time()-t2:.3f}sread,{t2
t1:.3f}swrite')
 file=stats_path.with_suffix('.json')
 t1=time.time()
 withopen(file,'w')asf:
 json.dump(stats,f) #savestats*.json
 t2=time.time()
 withopen(file)asf:
 x=json.load(f) #loadhypsdict
 print(f'stats.jsontimes:{time.time()-t2:.3f}sread,{t2
t1:.3f}swrite')
 #Save,printandreturn
 ifhub:
 print(f'Saving{stats_path.resolve()}...')
withopen(stats_path,'w')asf:
 json.dump(stats,f) #savestats.json
 ifverbose:
 print(json.dumps(stats,indent=2,sort_keys=False))
 returnstats
