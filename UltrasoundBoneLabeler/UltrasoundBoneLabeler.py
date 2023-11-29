import logging
import os
import qt
from typing import Annotated, Optional

import vtk
import cv2
import numpy as np

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    Minimum,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode, vtkMRMLVectorVolumeNode

from Logic import bone_probability_mapping, bone_surface_identification, files_manager


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
    autoCropping: bool = False
    resize: bool = False
    height: str = "256"
    width: str = "256"
    preprocessedVolume: vtkMRMLScalarVolumeNode
    inputVolume: vtkMRMLScalarVolumeNode
    gaussianKernelSize: Annotated[float, WithinRange(1, 63)] = 25
    binaryThreshold: Annotated[float, WithinRange(0, 1)] = 0.2
    transducerMargin: Annotated[float, WithinRange(0, 1)] = 0.1
    shadowSigma: Annotated[float, WithinRange(0, 200)] = 100
    localPhaseSigma: Annotated[float, WithinRange(0, 3)] = 0.5
    localPhaseWavelength: Annotated[float, WithinRange(0, 6)] = 2
    bestLineThreshold: Annotated[float, WithinRange(0, 1)] = 0.1
    bestLineCostFactor: Annotated[float, WithinRange(0, 1)] = 0.1
    LoGKernelSize: Annotated[float, WithinRange(1, 31)] = 31
    shadowNbSigmas: Annotated[float, WithinRange(0, 4)] = 2
    segmentationThickness: Annotated[float, WithinRange(1, 10)] = 3
    previewVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode


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
                
        if self._parameterNode.inputVolume:
            self.onAllVolumeButton()

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
        if self._parameterNode and self._parameterNode.inputVector:
            
            # Get the number of images in the volume
            array3D = slicer.util.arrayFromVolume(self._parameterNode.inputVector)
            numberOfSlices = array3D.shape[0]
            self.ui.rangeSlices.maximum = numberOfSlices
            # self.ui.rangeSlices.maximumValue = numberOfSlices
            
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
                             self._parameterNode.bestLineCostFactor,
                             int(self._parameterNode.segmentationThickness),
                             int(self.ui.rangeSlices.minimumValue),
                             int(self.ui.rangeSlices.maximumValue),
                             self._parameterNode.previewVolume,
                             self._parameterNode.outputSegmentation)
            
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
        # self.ui.rangeSlices.minimum = 
        # self.ui.rangeSlices.maximum = 
        layoutManager = slicer.app.layoutManager()
        red = layoutManager.sliceWidget("Red")
        redLogic = red.sliceLogic()
        redCurrentSlice = redLogic.GetSliceOffset()
        self.ui.rangeSlices.minimumValue = redCurrentSlice
        self.ui.rangeSlices.maximumValue = redCurrentSlice
        
        
    def onAllVolumeButton(self) -> None:
        """
        Run processing when user clicks "Preprocess" button.
        """
        self.ui.rangeSlices.minimumValue = self.ui.rangeSlices.minimum
        self.ui.rangeSlices.maximumValue = self.ui.rangeSlices.maximum

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
                           segmentName: str="Bone surface",
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
                bestLineCostFactor: float,
                segmentationThickness: int,
                startingSlice: int,
                endingSlice: int,
                previewVolume: vtkMRMLScalarVolumeNode,
                outputSegmentation: vtkMRMLSegmentationNode,
                segmentName: str="Bone surface") -> None:
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
        array3D = slicer.util.arrayFromVolume(inputVolume)[:, ::-1, ::-1]
        
        # Declare the algorithm classes
        boneProbMap = bone_probability_mapping.BoneProbabilityMapping(gaussianKernelSize,
                                                                      binaryThreshold,
                                                                      transducerMargin,
                                                                      LoGKernelSize,
                                                                      shadowSigma,
                                                                      shadowNbSigmas,
                                                                      localPhaseSigma,
                                                                      localPhaseWavelength)
        
        boneSurfId = bone_surface_identification.BoneSurfaceIdentification((15, 25),
                                                                           bestLineThreshold,
                                                                           bestLineCostFactor,
                                                                           segmentationThickness)
        
        # If a segmentation already exists, start from it
        segment = outputSegmentation.GetSegmentation().GetSegment(segmentName)
        if segment:
            label3D = slicer.util.arrayFromSegmentBinaryLabelmap(outputSegmentation, segmentName, inputVolume).astype(np.uint8)
            
        # Otherwise create a new one
        else:
            label3D = np.zeros(array3D.shape, dtype=np.uint8)
        
        # Apply the algorithm on every image
        for i, array2D in enumerate(array3D[startingSlice:endingSlice]):
            i = startingSlice+i
            label3D[i] = boneProbMap.apply_all_filters(array3D[i])
            label3D[i] = boneSurfId.identify_bone_surface(label3D[i])
            # arrayColor[i] = boneSurfId.draw_on_image(array3D[i], label3D[i], (0, 0, 255))
            
        # Update the volume node with the processed array
        slicer.util.updateVolumeFromArray(previewVolume, label3D[:, ::-1, ::-1])
        
        # Show the preview volume in slicer
        slicer.util.setSliceViewerLayers(background=inputVolume)
        
        # Actualize the segmentation
        self.createSegmentation(inputVolume, outputSegmentation, label3D[:, ::-1, ::-1], segmentName)
        
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
