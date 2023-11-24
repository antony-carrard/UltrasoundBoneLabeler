cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UltrasoundBoneLabeler)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/UltrasoundBoneLabeler")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Antony Carrard (HES-SO)")
set(EXTENSION_DESCRIPTION "An extension to semi-automatically segment bones on ultrasound images.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/UltrasoundBoneLabeler.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/UltrasoundBoneLabeler/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(UltrasoundBoneLabelerPy)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
