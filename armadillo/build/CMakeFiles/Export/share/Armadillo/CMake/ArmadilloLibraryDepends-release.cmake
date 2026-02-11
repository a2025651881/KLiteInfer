#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "armadillo" for configuration "Release"
set_property(TARGET armadillo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(armadillo PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libarmadillo.so.15.2.3"
  IMPORTED_SONAME_RELEASE "libarmadillo.so.15"
  )

list(APPEND _IMPORT_CHECK_TARGETS armadillo )
list(APPEND _IMPORT_CHECK_FILES_FOR_armadillo "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libarmadillo.so.15.2.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
