#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      Michael Laskey
# Created:     07/05/2017
#-------------------------------------------------------------------------------
from __future__ import division
from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk
import ttk, os, glob, random, IPython, cv2
import cPickle as pickle
import numpy as np
from fast_grasp_detect.configs.config import CONFIG
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

# colors for the bboxes
COLORS = ['red', 'blue', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 256, 256


class QueryLabeler():

    def __init__(self):
        # set up the main frame (Daniel: the standard way we start Tkinter)
        # http://effbot.org/tkinterbook/tkinter-hello-tkinter.htm
        self.parent = Tk()
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None
        self.currentLabelclass = ''
        self.cla_can_temp = []
        self.classcandidate_filename = 'class.txt'

        self.cfg = CONFIG()
        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.get_label)
        self.ldBtn.grid(row = 0, column = 2,sticky = W+E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
  
        # for ease of use
        self.parent.bind("f", lambda event: self.delBBox())
        self.parent.bind("q", lambda event: self.class_key_update("q"))
        self.parent.bind("w", lambda event: self.class_key_update("w"))
        self.parent.bind("e", lambda event: self.class_key_update("e"))
        self.parent.bind("r", lambda event: self.class_key_update("r"))
        # self.parent.bind("t", lambda event: self.class_key_update("t"))

        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # choose class (Daniel: https://docs.python.org/2/library/ttk.html)
        self.classname = StringVar()
        self.classcandidate = ttk.Combobox(self.frame, state='readonly', textvariable=self.classname)
        self.classcandidate.grid(row=1,column=2)
        self.cla_can_temp = self.cfg.CLASSES
        #print("QueryLabeler.__init__(). Classes in it: {}".format(self.cla_can_temp))

        self.classcandidate['values'] = self.cla_can_temp
        self.classcandidate.current(0)
        self.currentLabelclass = self.classcandidate.get() #init
        self.btnclass = Button(self.frame, text = 'ComfirmClass', command = self.setClass)
        self.btnclass.grid(row=2,column=2,sticky = W+E)
        #print("QueryLabeler.__init__(). self.classcandidate.get(): {}".format(self.classcandidate.get()))

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 3, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 4, column = 2, sticky = N+S)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 5, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 6, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 7, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Send Command', width = 10, command = self.sendCommand)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        self.online = True

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        self.label_count = 0


    def loadDir(self, dbg = False):
        s = 000
        # get image list
        self.imageDir = self.cfg.IMAGE_PATH
        # print self.imageDir
        #print self.category

        self.imageList = glob.glob(os.path.join(self.imageDir, '*.png'))
        # print self.imageList
        if len(self.imageList) == 0:
            print 'No .JPG images found in the specified dir!'
            return

        #get label list
        self.labelDir = self.cfg.LABEL_PATH
        self.labelListOrig = glob.glob(os.path.join(self.labelDir, '*.p'))
        self.labelList = [os.path.split(label)[-1].split('.')[0] for label in self.labelListOrig]

        #remove already labeled images
        self.imageList = [img for img in self.imageList if os.path.split(img)[-1].split('.')[0] not in self.labelList]
        #to review already labelled ones
        # self.imageList = [img for img in self.imageList if os.path.split(img)[-1].split('.')[0] in self.labelList]

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)
        # set up output dir
        self.outDir = self.cfg.LABEL_PATH

        # load example bboxes
        #self.egDir = os.path.join(r'./Examples', '%03d' %(self.category))
        self.egDir = os.path.join(r'./Examples/demo')
        if not os.path.exists(self.egDir):
            return
        filelist = glob.glob(os.path.join(self.egDir, '*.png'))
        self.tmp = []
        self.egList = []
        random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])

        self.loadImage()
        print '%d images loaded from %s' %(self.total, s)


    def get_label(self):
        if self.image is None:
            self.current_image = self.cam.read_color_data()
            tmp_depth = self.cam.read_depth_data()
        else: 
            self.current_image = self.image

        if False:
            # ----------------------------------------------------------------------
            # Temporary debugging to get viewpoints to align w/H's data, as needed.
            # Will save the image(s) based on what the robot sees during the click interface.
            # For depth cutoff, note that the HSR uses *millimeters*.
            # ----------------------------------------------------------------------
            kk = len([x for x in os.listdir('imgs/') if 'view_close_rgb' in x])
            img_name = 'imgs/view_close_rgb_{}.png'.format(str(kk).zfill(2))
            cv2.imwrite(img_name, self.current_image)

            kk = len([x for x in os.listdir('imgs/') if 'view_close_depth' in x])
            img_name = 'imgs/view_close_depth_{}.png'.format(str(kk).zfill(2))

            # save tmp_depth to look at histograms later, etc.
            numpy_name = 'temp_depth_{}.txt'.format(kk)
            np.savetxt(numpy_name, tmp_depth)
            print("just saved {}".format(numpy_name))

            d_img = depth_to_net_dim(tmp_depth, cutoff=1250)
            cv2.imwrite(img_name, d_img)
            # ----------------------------------------------------------------------

        self.tkimg = ImageTk.PhotoImage(Image.fromarray(self.current_image))
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
        # load labels
        self.clearBBox()
        self.imagename = 'frame_'+str(self.label_count)
        labelname = self.imagename + '.p'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        self.lock = True
        

    def loadImage(self):
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)

        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.p'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        # print('testing loading of label')
        # print(self.labelfilename)
        # print(os.path.exists(self.labelfilename))
        if os.path.exists(self.labelfilename):
            label = pickle.load( open(self.labelfilename, "rb") )
            for obj in label['objects']:
                curr_bbox = list(obj['box_index'])
                curr_bbox.append(self.cfg.CLASSES[obj['num_class_label']])
                tmp = curr_bbox
                self.bboxList.append(tuple(curr_bbox))
                tmpId = self.mainPanel.create_rectangle(int(tmp[0]), int(tmp[1]), \
                                                        int(tmp[2]), int(tmp[3]), \
                                                        width = 2, \
                                                        outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                # print tmpId
                self.bboxIdList.append(tmpId)
                #print("in `QueryLabeler.loadImage()` putting listbox with class {}".format(tmp[4]))
                self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' %(tmp[4],int(tmp[0]), int(tmp[1]), \
                                                                  int(tmp[2]), int(tmp[3])))
                self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
          

    def saveImage(self):
        """Updates `self.label_data` which is provided to the online labeler
        when we do data collection for the bed. 
        
        Specifically, returns a dictionary s.t. label_data['objects'] is a list
        of dictionaries, each dict representring one of the bounding boxes we've
        drawn (usually only one). See the config file:

        https://github.com/DanielTakeshi/fast_grasp_detect/blob/master/src/fast_grasp_detect/configs/config.py

        for what these mean, but the classes here originally represented "yes"
        or "no" but now they are four things ... but in the `check_success`
        file, we call `if (result['class'] == 0)` for success, so as long as the
        _index_ of the class from `bbox[4]` is 0, we know it's a success and
        then we can transition the HSR to the other side.

        For bboxList, it gets appended from `loadImage()` and `mouseClick()`.
        It's clear that for both, the format is (x1,y1,x2,y2,class_label).
        """
        label_data = {}
        label_data['num_labels'] = len(self.bboxList)
        objects = []
        for bbox in self.bboxList:
            obj = {}
            obj['box'] = bbox[0:4]
            obj['class'] = self.cfg.CLASSES.index(bbox[4])
            objects.append(obj)
        label_data['objects'] = objects
        self.label_data = label_data
        print("QueryLabeler.saveImage(), image no. {} saved, objects: {}".format(self.cur, objects))


    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2, self.currentLabelclass))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '%s : (%d, %d) -> (%d, %d)' %(self.currentLabelclass,x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']


    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)])


    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0


    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)


    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []


    def prevImage(self, event = None):
        #print("QueryLabeler.prevImage(), self.cur {}, event {}, calling `saveImage()` ...".format(self.cur, event))
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()


    def sendCommand(self, event = None):
        #print("QueryLabeler.sendCommand(), self.cur {}, event {}, calling `saveImage()` ...".format(self.cur, event))
        self.saveImage()
        self.lock = False
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()


    def gotoImage(self):
        #print("QueryLabeler.gotoImage(), calling `saveImage()` ...")
        idx = int(self.idxEntry.get())
        if 1 <= idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()


    def setClass(self):
        """Originally an outdated mapping. Use index 0 to indicate a SUCCESS,
        anything else is a failure.
        """
        #print("In QueryLabeler.setClass(), starting `self.currentLabelclass`: {}".format(self.currentLabelclass))
        self.currentLabelclass = self.classcandidate.get()
        #mapping = {"q": ("grasp", 0), "w": ("singulate", 1), "e": ("suction", 2), "r": ("quit",3)}
        mapping = {"q": ("success", 0), "w": ("failure", 1), "e": ("failure", 2), "r": ("failure",3)}
        self.currentLabelclass = mapping[class_label][0]
        self.classcandidate.current(mapping[class_label][1])
        #print("    now, self.currentLabelClass: {}".format(self.currentLabelclass))
        #print("    now, class_label: {}".format(class_label))


    def run(self, cam, image=None):
        #self.parent.resizable(width=True, height=True)
        self.image = image
        self.cam = cam
        #self.current_image = img
        self.parent.mainloop()


if __name__ == '__main__':
    root = Tk()
    #tool = LabelTool(root) # Daniel: ? doesn't seem to be in TKinter library.
    root.resizable(width=True, height=True)
    root.mainloop()
