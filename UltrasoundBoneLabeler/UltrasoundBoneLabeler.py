import logging
import os
import qt
from typing import Annotated, Optional
from enum import Enum

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode, vtkMRMLVectorVolumeNode

try:
    import vtk
    import cv2
    import numpy as np
    import re
except:
    slicer.util.pip_install("numpy")
    slicer.util.pip_install("opencv-python")
    slicer.util.pip_install("vtk")
    slicer.util.pip_install("regex")
finally:
    import vtk
    import cv2
    import numpy as np
    import re

from Logic import bone_probability_mapping, bone_surface_identification, files_manager

# Default parameters constants
GAUSSIAN_KERNEL_SIZE = 25
BINARY_THRESHOLD = 0.2
TRANSDUCER_MARGIN = 0.1
SHADOW_SIGMA = 100
LOCAL_PHASE_SIGMA = 0.5
LOCAL_PHASE_WAVELENGTH = 150
BEST_LINE_THRESHOLD = 0.02
BEST_LINE_SIGMA = 5
LOG_KERNEL_SIZE = 31
SHADOW_NB_SIGMAS = 2
SEGMENTATION_THICKNESS = 5
MINIMUM_BONE_WIDTH = 0.3

SEGMENT_NAME = "Bone surface"

class FileExtensions(Enum):
    JPG = ".jpg"
    NPY = ".npy"
    PNG = ".png"

#
# UltrasoundBoneLabeler
#

class UltrasoundBoneLabeler(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "UltrasoundBoneLabeler"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Antony Carrard (HES-SO)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an extension designed to semi-automatically segment bones on ultrasound images.
See more information in <a href="https://github.com/antony-carrard/UltrasoundBoneLabeler">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Antony Carrard.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # UltrasoundBoneLabeler1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='UltrasoundBoneLabeler',
        sampleName='UltrasoundBoneLabeler1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'UltrasoundBoneLabeler1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='UltrasoundBoneLabeler1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='UltrasoundBoneLabeler1'
    )

    # UltrasoundBoneLabeler2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='UltrasoundBoneLabeler',
        sampleName='UltrasoundBoneLabeler2',
        thumbnailFileName=os.path.join(iconsPath, 'UltrasoundBoneLabeler2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='UltrasoundBoneLabeler2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='UltrasoundBoneLabeler2'
    )


#
# UltrasoundBoneLabelerParameterNode
#

@parameterNodeWrapper
class UltrasoundBoneLabelerParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """
    inputVector: vtkMRMLVectorVolumeNode
    autoCropping: bool = True
    resize: bool = True
    height: str = "256"
    width: str = "256"
    preprocessedVolume: vtkMRMLScalarVolumeNode
    inputVolume: vtkMRMLScalarVolumeNode
    gaussianKernelSize: Annotated[float, WithinRange(1, 63)] = GAUSSIAN_KERNEL_SIZE
    binaryThreshold: Annotated[float, WithinRange(0, 1)] = BINARY_THRESHOLD
    transducerMargin: Annotated[float, WithinRange(0, 1)] = TRANSDUCER_MARGIN
    shadowSigma: Annotated[float, WithinRange(1, 200)] = SHADOW_SIGMA
    localPhaseSigma: Annotated[float, WithinRange(0.1, 3)] = LOCAL_PHASE_SIGMA
    localPhaseWavelength: Annotated[float, WithinRange(10, 500)] = LOCAL_PHASE_WAVELENGTH
    bestLineThreshold: Annotated[float, WithinRange(0, 1)] = BEST_LINE_THRESHOLD
    bestLineSigma: Annotated[float, WithinRange(1, 40)] = BEST_LINE_SIGMA
    LoGKernelSize: Annotated[float, WithinRange(1, 31)] = LOG_KERNEL_SIZE
    shadowNbSigmas: Annotated[float, WithinRange(1, 4)] = SHADOW_NB_SIGMAS
    segmentationThickness: Annotated[float, WithinRange(1, 10)] = SEGMENTATION_THICKNESS
    minimumBoneWidth: Annotated[float, WithinRange(0, 1)] = MINIMUM_BONE_WIDTH
    previewVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode
    volumeToExport: vtkMRMLScalarVolumeNode
    fileNameImages: str
    includeEmptyImages: bool
    segmentationToExport: vtkMRMLSegmentationNode
    fileNameLabels: str
    includeEmptyLabels: bool


#
# UltrasoundBoneLabelerWidget
#

class UltrasoundBoneLabelerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/UltrasoundBoneLabeler.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = UltrasoundBoneLabelerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.preprocessButton.connect('clicked(bool)', self.onPreprocessButton)
        self.ui.currentSliceButton.connect('clicked(bool)', self.onCurrentSliceButton)
        self.ui.allVolumeButton.connect('clicked(bool)', self.onAllVolumeButton)
        self.ui.defaultParametersButton.connect('clicked(bool)', self.onDefaultParametersButton)
        self.ui.previewButton.connect('clicked(bool)', self.onPreviewButton)
        self.ui.exportImagesButton.connect('clicked(bool)', self.onExportImagesButton)
        self.ui.exportLabelsButton.connect('clicked(bool)', self.onExportLabelsButton)
        
        # Comboboxes
        self.ui.inputVector.connect('currentNodeChanged(bool)', self.onInputVectorModification)
        self.ui.inputVolume.connect('currentNodeChanged(bool)', self.onInputVolumeModification)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPreprocess)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._volumeLoaded)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
                
        if not self._parameterNode.inputVector:
            firstVectorNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLVectorVolumeNode")
            if firstVectorNode:
                self._parameterNode.inputVector = firstVectorNode
                
        # Create a new volume for the preprocessed volume
        if not self._parameterNode.preprocessedVolume:
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Preprocessed Volume")
            self._parameterNode.preprocessedVolume = volumeNode
                
        # Create a new volume for the preview
        if not self._parameterNode.previewVolume:
            volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Preview Volume")
            self._parameterNode.previewVolume = volumeNode
            self._parameterNode.volumeToExport = volumeNode
        
        # If no segmentation exists, create a new one
        if not self._parameterNode.outputSegmentation:
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Segmentation")
            self._parameterNode.outputSegmentation = segmentationNode
            self._parameterNode.segmentationToExport = segmentationNode
            
        # Actualize the slider of the images selection
        if self._parameterNode.inputVolume:
            self._volumeLoaded()
            
        # Actualize the file names to export if empty. Take the name of the inputVector without the extension.
        if not self._parameterNode.fileNameImages:
            fullFilename = self._parameterNode.inputVector.GetName()
            filename = re.search("^[^.]*", fullFilename).group()
            if filename:
                self._parameterNode.fileNameImages = filename
                self._parameterNode.fileNameLabels = filename
            
        # Populate the comboBoxes
        if self.ui.imagesTypeComboBox.count == 0:
            fileTypeImages = [FileExtensions.JPG.value, FileExtensions.PNG.value]
            self.ui.imagesTypeComboBox.addItems(fileTypeImages)
            self.ui.imagesTypeComboBox.setCurrentIndex(1)   # .png by default
        
        if self.ui.labelsTypeComboBox.count == 0:
            fileTypeLabels = [FileExtensions.NPY.value, FileExtensions.PNG.value]
            self.ui.labelsTypeComboBox.addItems(fileTypeLabels)
            self.ui.labelsTypeComboBox.setCurrentIndex(1)   # .png by default

    def setParameterNode(self, inputParameterNode: Optional[UltrasoundBoneLabelerParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPreprocess)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._volumeLoaded)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanPreprocess)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._volumeLoaded)
            self._checkCanApply()
            self._checkCanPreprocess()
            self._volumeLoaded()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.previewVolume:
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False
            
    def _checkCanPreprocess(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVector and self._parameterNode.preprocessedVolume:
            self.ui.preprocessButton.toolTip = "Preprocess input volume"
            self.ui.preprocessButton.enabled = True
        else:
            self.ui.preprocessButton.toolTip = "Select input and output options"
            self.ui.preprocessButton.enabled = False
            
    def _volumeLoaded(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            
            # Get the number of images in the volume
            array3D = slicer.util.arrayFromVolume(self._parameterNode.inputVolume)
            numberOfSlices = array3D.shape[0]
            if self.ui.rangeSlices.maximum != numberOfSlices:
                self.ui.rangeSlices.maximum = numberOfSlices
                self.ui.rangeSlices.maximumValue = numberOfSlices
            
            # Activate the slices buttons
            self.ui.currentSliceButton.enabled = True
            self.ui.allVolumeButton.enabled = True
        else:
            self.ui.currentSliceButton.enabled = False
            self.ui.allVolumeButton.enabled = False

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.apply(self._parameterNode.inputVolume,
                             int(self._parameterNode.gaussianKernelSize),
                             self._parameterNode.binaryThreshold,
                             self._parameterNode.transducerMargin,
                             int(self._parameterNode.LoGKernelSize),
                             self._parameterNode.shadowSigma,
                             self._parameterNode.shadowNbSigmas,
                             self._parameterNode.localPhaseSigma,
                             self._parameterNode.localPhaseWavelength,
                             self._parameterNode.bestLineThreshold,
                             self._parameterNode.bestLineSigma,
                             self._parameterNode.minimumBoneWidth,
                             int(self._parameterNode.segmentationThickness),
                             int(self.ui.rangeSlices.minimumValue),
                             int(self.ui.rangeSlices.maximumValue),
                             self._parameterNode.previewVolume,
                             self._parameterNode.outputSegmentation,
                             self.getPreviewButtons())
            
    def onPreprocessButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Compute output
            self.logic.preprocess(self._parameterNode.inputVector,
                                  self._parameterNode.preprocessedVolume,
                                  self._parameterNode.autoCropping,
                                  self._parameterNode.resize,
                                  int(self._parameterNode.height),
                                  int(self._parameterNode.width))
            
            # Refresh the input volume selector
            self._parameterNode.inputVolume = self._parameterNode.preprocessedVolume
            
    def onCurrentSliceButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        layoutManager = slicer.app.layoutManager()
        red = layoutManager.sliceWidget("Red")
        redLogic = red.sliceLogic()
        redCurrentSlice = redLogic.GetSliceOffset()
        self.ui.rangeSlices.minimumValue = redCurrentSlice
        self.ui.rangeSlices.maximumValue = redCurrentSlice
        # Do the instruction once more in case the first one failed
        self.ui.rangeSlices.minimumValue = redCurrentSlice
        self.ui.rangeSlices.maximumValue = redCurrentSlice
        
    def onAllVolumeButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        self.ui.rangeSlices.minimumValue = self.ui.rangeSlices.minimum
        self.ui.rangeSlices.maximumValue = self.ui.rangeSlices.maximum
        # Do the instruction once more in case the first one failed
        self.ui.rangeSlices.minimumValue = self.ui.rangeSlices.minimum
        self.ui.rangeSlices.maximumValue = self.ui.rangeSlices.maximum
        
    def onDefaultParametersButton(self) -> None:
        """
        Run processing when user clicks "DefaultParameters" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            
            self._parameterNode.gaussianKernelSize = GAUSSIAN_KERNEL_SIZE
            self._parameterNode.binaryThreshold = BINARY_THRESHOLD
            self._parameterNode.transducerMargin = TRANSDUCER_MARGIN
            self._parameterNode.shadowSigma = SHADOW_SIGMA
            self._parameterNode.localPhaseSigma = LOCAL_PHASE_SIGMA
            self._parameterNode.localPhaseWavelength = LOCAL_PHASE_WAVELENGTH
            self._parameterNode.bestLineThreshold = BEST_LINE_THRESHOLD
            self._parameterNode.bestLineSigma = BEST_LINE_SIGMA
            self._parameterNode.LoGKernelSize = LOG_KERNEL_SIZE
            self._parameterNode.shadowNbSigmas = SHADOW_NB_SIGMAS
            self._parameterNode.segmentationThickness = SEGMENTATION_THICKNESS
            self._parameterNode.minimumBoneWidth = MINIMUM_BONE_WIDTH
            self.onAllVolumeButton()
            
    def getPreviewButtons(self) -> list:
        """
        Get all the preview button in a list for simplicity
        """
        return [self.ui.radioButton,
                self.ui.radioButton1,
                self.ui.radioButton2,
                self.ui.radioButton3,
                self.ui.radioButton4,
                self.ui.radioButton5,
                self.ui.radioButton6,
                self.ui.radioButton7,
                self.ui.radioButton8,
                self.ui.radioButton9,
                self.ui.radioButton10,
                self.ui.radioButton11,
                self.ui.radioButton12]
            
    def onPreviewButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            # Send the radio button to the function
            previewButtons = self.getPreviewButtons()
            self.logic.showVolume(previewButtons, self._parameterNode.previewVolume)
            
    def onExportImagesButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            
            self.logic.exportImages(self._parameterNode.volumeToExport,
                                    self._parameterNode.segmentationToExport,
                                    self.ui.imagesDirectoryButton.directory,
                                    self._parameterNode.fileNameImages,
                                    self.ui.imagesTypeComboBox.currentText,
                                    self._parameterNode.includeEmptyImages)
            
    def onExportLabelsButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            
            self.logic.exportLabels(self._parameterNode.volumeToExport,
                                    self._parameterNode.segmentationToExport,
                                    self.ui.labelsDirectoryButton.directory,
                                    self._parameterNode.fileNameLabels,
                                    self.ui.labelsTypeComboBox.currentText,
                                    self._parameterNode.includeEmptyLabels)
            
    def onInputVectorModification(self) -> None:
        # Actualize the file names to export if empty
        if not self._parameterNode.fileNameImages:
            self._parameterNode.fileNameImages = self._parameterNode.inputVector.GetName()
            self._parameterNode.fileNameLabels = self._parameterNode.inputVector.GetName()
            
    def onInputVolumeModification(self) -> None:
        # Actualize the slider of the images selection
        if self._parameterNode.inputVolume:
            self.onAllVolumeButton()

#
# UltrasoundBoneLabelerLogic
#

class UltrasoundBoneLabelerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return UltrasoundBoneLabelerParameterNode(super().getParameterNode())
        
    def createSegmentation(self,
                           inputVolume: vtkMRMLScalarVolumeNode,
                           outputSegmentation: vtkMRMLSegmentationNode,
                           label3D: np.ndarray,
                           segmentName: str=SEGMENT_NAME,
                           segmentColor: tuple[float, float, float]=(1, 0, 0)) -> None:
        
        # First, check if the current segment exists
        segment = outputSegmentation.GetSegmentation().GetSegment(segmentName)
        
        # If not, create it
        if not segment:
            outputSegmentation.GetSegmentation().AddEmptySegment(segmentName)
            segment = outputSegmentation.GetSegmentation().GetSegment(segmentName)
            
        # Color the segment
        segment.SetColor(segmentColor)
        
        # Apply the 3D label to the segment
        slicer.util.updateSegmentBinaryLabelmapFromArray(narray=label3D,
                                                         segmentationNode=outputSegmentation,
                                                         segmentId=segmentName,
                                                         referenceVolumeNode=inputVolume)
        
    def showVolume(self, previewButtons, previewVolume) -> None:
        if previewButtons[0].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.array3D[:, ::-1, ::-1])
        if previewButtons[1].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.gaussian3D[:, ::-1, ::-1])
        if previewButtons[2].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.mask3D[:, ::-1, ::-1])
        if previewButtons[3].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.LoG3D[:, ::-1, ::-1])
        if previewButtons[4].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.Shadow3D[:, ::-1, ::-1])
        if previewButtons[5].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.energy3D[:, ::-1, ::-1])
        if previewButtons[6].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.phase3D[:, ::-1, ::-1])
        if previewButtons[7].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.symmetry3D[:, ::-1, ::-1])
        if previewButtons[8].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.ibs3D[:, ::-1, ::-1])
        if previewButtons[9].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.probMap3D[:, ::-1, ::-1])
        if previewButtons[10].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.contour3D[:, ::-1, ::-1])
        if previewButtons[11].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.label3D[:, ::-1, ::-1])
        if previewButtons[12].isChecked():
            slicer.util.updateVolumeFromArray(previewVolume, self.tracedLabel3D[:, ::-1, ::-1])
                
        # Show the preview volume in slicer
        slicer.util.setSliceViewerLayers(background=previewVolume)
        
        
    def apply(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                gaussianKernelSize: int,
                binaryThreshold: float,
                transducerMargin: float,
                LoGKernelSize: int,
                shadowSigma: float,
                shadowNbSigmas: float,
                localPhaseSigma: float,
                localPhaseWavelength: float,
                bestLineThreshold: float,
                bestLineSigma: float,
                minimumBoneWidth: float,
                segmentationThickness: int,
                startingSlice: int,
                endingSlice: int,
                previewVolume: vtkMRMLScalarVolumeNode,
                outputSegmentation: vtkMRMLSegmentationNode,
                previewButtons: list,
                segmentName: str=SEGMENT_NAME) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param showResult: show output volume in slice viewers
        """

        # if not inputVolume or not outputVolume:
        #     raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Get the 3D numpy array from the slicer volume
        self.array3D = slicer.util.arrayFromVolume(inputVolume)[:, ::-1, ::-1]
        
        # Declare the algorithm classes
        numberOfImages = self.array3D[0].shape
        self.boneProbMap = bone_probability_mapping.BoneProbabilityMapping(numberOfImages,
                                                                      gaussianKernelSize,
                                                                      binaryThreshold,
                                                                      transducerMargin,
                                                                      LoGKernelSize,
                                                                      shadowSigma,
                                                                      shadowNbSigmas,
                                                                      localPhaseSigma,
                                                                      localPhaseWavelength)
        
        boneSurfId = bone_surface_identification.BoneSurfaceIdentification(bestLineThreshold,
                                                                           bestLineSigma,
                                                                           minimumBoneWidth,
                                                                           segmentationThickness)
        
        # If a segmentation already exists, start from it
        segment = outputSegmentation.GetSegmentation().GetSegment(segmentName)
        if segment:
            self.tracedLabel3D = slicer.util.arrayFromSegmentBinaryLabelmap(outputSegmentation, segmentName, inputVolume).astype(np.uint8)[:, ::-1, ::-1]
            
            
        # Instanciate the arrays if they are not already existing
        try:
            self.gaussian3D
        except:
            pass
        else:
            self.gaussian3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.mask3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.LoG3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.Shadow3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.energy3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.phase3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.symmetry3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.ibs3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.probMap3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.contour3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.label3D = np.zeros(self.array3D.shape, dtype=np.uint8)
            self.tracedLabel3D = np.zeros(self.array3D.shape, dtype=np.uint8)
        
        # Apply the algorithm on every image
        for i in range(len(self.array3D[startingSlice:endingSlice+1])):
            i = startingSlice+i
            self.gaussian3D[i], self.mask3D[i], self.LoG3D[i], self.Shadow3D[i], self.energy3D[i], self.phase3D[i], self.symmetry3D[i], self.ibs3D[i], self.probMap3D[i] = self.boneProbMap.apply_all_filters(self.array3D[i])
            self.contour3D[i], self.label3D[i], self.tracedLabel3D[i] = boneSurfId.identify_bone_surface(self.probMap3D[i])
            
        # Update the volume node with the processed array
        self.showVolume(previewButtons, previewVolume)
        
        # Actualize the segmentation
        self.createSegmentation(inputVolume, outputSegmentation, self.tracedLabel3D[:, ::-1, ::-1], segmentName)
        
        # Show the segmentation in 3D
        outputSegmentation.CreateClosedSurfaceRepresentation()

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
        
        # Return the binary labeled array
        # return label3D.astype(bool).astype(int)

    def preprocess(self,
                   inputVector: vtkMRMLVectorVolumeNode,
                   preprocessedVolume: vtkMRMLScalarVolumeNode,
                   autoCropping: bool = False,
                   resize: bool = False,
                   height: int = 256,
                   width: int = 256) -> None:
        
        fm = files_manager.FileManager()
        
        # Get the 4D numpy array from the slicer vector
        array4D = slicer.util.arrayFromVolume(inputVector)
        
        # Get the gray part of the vector
        array3D = array4D[:, :, :, 0]
        
        # Resize and crop the volume
        if autoCropping:
            array3D = fm.auto_crop(array3D)
        if resize:
            array3D = fm.resize_images(array3D, (height, width))
        
        # Update the volume with the resized array
        slicer.util.updateVolumeFromArray(preprocessedVolume, array3D[:, ::-1, ::-1])
        slicer.util.setSliceViewerLayers(preprocessedVolume)
        
        # Reset the field of view
        slicer.util.resetSliceViews()
        
    def exportImages(self,
                    volumeToExport: vtkMRMLScalarVolumeNode,
                    currentSegmentation: vtkMRMLSegmentationNode,
                    currentPath: str,
                    fileName: str,
                    imagesType: str,
                    includeEmptyImages: bool,
                    segmentName: str=SEGMENT_NAME) -> None:
        
        # Get the 3D numpy array from the slicer volume
        arrayToExport = slicer.util.arrayFromVolume(volumeToExport)[:, ::-1, ::-1]
        
        # Get the segmentation from slicer
        segmentation = slicer.util.arrayFromSegmentBinaryLabelmap(currentSegmentation, segmentName, volumeToExport).astype(np.uint8)[:, ::-1, ::-1]
            
        # Iterate on the volume to export
        for i in range(len(arrayToExport)):
            
            # If the option is set to ignore empty segmentation, skip the image to save
            if not includeEmptyImages and not segmentation[i].any():
                continue
            
            # Else, save the image with the number of the image at the end of the file
            filenameWithNumber = f"{fileName}_{i}"
            filenameWithExtension = filenameWithNumber + imagesType
            fullFilename = os.path.join(currentPath, filenameWithExtension)
            
            # Save the image
            cv2.imwrite(fullFilename, arrayToExport[i])
            
            # print out the info
            logging.info(f'Image {filenameWithExtension} successfully saved in directory {currentPath}.')

    def exportLabels(self,
                    referenceVolume: vtkMRMLScalarVolumeNode,
                    segmentationToExport: vtkMRMLSegmentationNode,
                    currentPath: str,
                    fileName: str,
                    LabelsType: str,
                    includeEmptyLabels: bool,
                    segmentName: str=SEGMENT_NAME) -> None:
        
        # Get the segmentation from slicer
        segmentationArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationToExport, segmentName, referenceVolume).astype(np.uint8)[:, ::-1, ::-1]
            
        # Iterate on the segmentation to export
        for i in range(len(segmentationArray)):
            
            # If the option is set to ignore empty segmentation, skip the image to save
            if not includeEmptyLabels and not segmentationArray[i].any():
                continue
            
            # Else, save the image with the number of the image at the end of the file
            filenameWithNumber = f"{fileName}_{i}"
            filenameWithExtension = filenameWithNumber + LabelsType
            fullFilename = os.path.join(currentPath, filenameWithExtension)
            
            # If a NumPy format is specified, save the image as a NumPy array
            if LabelsType == FileExtensions.NPY.value:
                np.save(fullFilename, segmentationArray[i])
                
            # Otherwise save the image normally
            else:
                # To make the masks visible in the file explorer, multiply the values by the max value of uint8 (255).
                # This can be removed for multiple classes as it could cause overflow issues.
                segmentationArray[i] *= np.iinfo(np.uint8).max
                
                # Save the image
                cv2.imwrite(fullFilename, segmentationArray[i])
            
            # print out the info
            logging.info(f'Image {filenameWithExtension} successfully saved in directory {currentPath}.')

#
# UltrasoundBoneLabelerTest
#

class UltrasoundBoneLabelerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        # self.test_UltrasoundBoneLabeler1()
        self.test_fromFile()

    def test_UltrasoundBoneLabeler1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('UltrasoundBoneLabeler1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = UltrasoundBoneLabelerLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
        
        
    def test_fromFile(self):
        
        self.delayDisplay("Starting the test")
        
        # inputVolume = slicer.util.loadVolume('C:/Users/Antony/OneDrive/HES-SO/Semestre2/PA_UltraMotion/Data/day1/3DUS_L_probe1_conf1_ss1.dcm')
        
        # logic = UltrasoundBoneLabelerLogic()
        
        # label3D = logic.applyFilters(inputVolume, False)
        # logic.createSegmentation(inputVolume, label3D)
