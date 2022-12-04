/*=========================================================================

Program:   Visualization Toolkit
Module:    vtkSetGet.h

Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME SetGet Macros - standard macros for setting/getting instance variables
// .SECTION Description
// The SetGet macros are used to interface to instance variables
// in a standard fashion. This includes properly treating modified time
// and printing out debug information.
//
// Macros are available for built-in types; for character strings;
// vector arrays of built-in types size 2,3,4; for setting objects; and
// debug, warning, and error printout information.

#ifndef SETGET_H
#define SETGET_H

#include <math.h>

// Convert a macro representing a value to a string.
//
// Example: vtkQuoteMacro(__LINE__) will expand to "1234" whereas
// vtkInternalQuoteMacro(__LINE__) will expand to "__LINE__"
#define InternalQuoteMacro(x) #x
#define QuoteMacro(x) InternalQuoteMacro(x)

//
// Set built-in type.  Creates member Set"name"() (e.g., SetVisibility());
//
#define SetMacro(name,type) \
virtual void Set##name (type _arg) \
    { \
    this->name = _arg; \
    }

//
// Get built-in type.  Creates member Get"name"() (e.g., GetVisibility());
//
#define GetMacro(name,type) \
virtual type Get##name () { \
  return this->name; \
    }


//
// Set character string.  Creates member Set"name"()
// (e.g., SetFilename(char *));
//
#define SetStringMacro(name) \
virtual void Set##name (const char* _arg) \
    { \
  if ( this->name == NULL && _arg == NULL) { return;} \
  if ( this->name && _arg && (!strcmp(this->name,_arg))) { return;} \
  delete [] this->name; \
  if (_arg) \
      { \
    size_t n = strlen(_arg) + 1; \
    char *cp1 =  new char[n]; \
    const char *cp2 = (_arg); \
    this->name = cp1; \
    do { *cp1++ = *cp2++; } while ( --n ); \
      } \
      else \
	    { \
    this->name = NULL; \
	    } \
    }

//
// Get character string.  Creates member Get"name"()
// (e.g., char *GetFilename());
//
#define GetStringMacro(name) \
virtual char* Get##name () { \
  return this->name; \
    }

//
// Set built-in type where value is constrained between min/max limits.
// Create member Set"name"() (eg., SetRadius()). #defines are
// convenience for clamping open-ended values.
// The Get"name"MinValue() and Get"name"MaxValue() members return the
// min and max limits.
//
#define SetClampMacro(name,type,min,max) \
virtual void Set##name (type _arg) \
    { \
    this->name = (_arg<min?min:(_arg>max?max:_arg)); \
    } \
virtual type Get##name##MinValue () \
    { \
  return min; \
    } \
virtual type Get##name##MaxValue () \
    { \
  return max; \
    }

//
// This macro defines a body of set object macro. It can be used either in
// the header file vtkSetObjectMacro or in the implementation one
// vtkSetObjectMacro. It sets the pointer to object; uses vtkObject
// reference counting methodology. Creates method
// Set"name"() (e.g., SetPoints()).
//
#define SetObjectBodyMacro(name,type,args)                   \
    {                                                             \
  if (this->name != args)                                       \
      {                                                           \
    type* tempSGMacroVar = this->name;                          \
    this->name = args;                                          \
    if (this->name != NULL) { this->name->Register(this); }     \
    if (tempSGMacroVar != NULL)                                 \
	      {                                                         \
      tempSGMacroVar->UnRegister(this);                         \
	      }                                                         \
    this->Modified();                                           \
      }                                                           \
    }

//
// Set pointer to object; uses vtkObject reference counting methodology.
// Creates method Set"name"() (e.g., SetPoints()). This macro should
// be used in the header file.
//
#define SetObjectMacro(name,type)            \
virtual void Set##name (type* _arg)             \
    {                                             \
  SetObjectBodyMacro(name,type,_arg);        \
    }

//
// Set pointer to object; uses vtkObject reference counting methodology.
// Creates method Set"name"() (e.g., SetPoints()). This macro should
// be used in the implementation file. You will also have to write
// prototype in the header file. The prototype should look like this:
// virtual void Set"name"("type" *);
//
// Please use vtkCxxSetObjectMacro not vtkSetObjectImplementationMacro.
// The first one is just for people who already used it.
#define SetObjectImplementationMacro(class,name,type)        \
  CxxSetObjectMacro(class,name,type)

#define CxxSetObjectMacro(class,name,type)   \
void class::Set##name (type* _arg)              \
    {                                             \
  SetObjectBodyMacro(name,type,_arg);        \
    }

//
// Get pointer to object wrapped in vtkNew.  Creates member Get"name"
// (e.g., GetPoints()).  This macro should be used in the header file.
//
#define GetNewMacro(name,type)                                    \
virtual type *Get##name ()                                              \
    {                                                                     \
  DebugMacro(<< this->GetClassName() << " (" << this                 \
                << "): returning " #name " address "                    \
                << this->name.GetPointer() );                           \
  return this->name.GetPointer();                                       \
    }

//
// Get pointer to object.  Creates member Get"name" (e.g., GetPoints()).
// This macro should be used in the header file.
//
#define GetObjectMacro(name,type)                                    \
virtual type *Get##name ()                                              \
    {                                                                     \
  DebugMacro(<< this->GetClassName() << " (" << this                 \
                << "): returning " #name " address " << this->name );   \
  return this->name;                                                    \
    }

//
// Create members "name"On() and "name"Off() (e.g., DebugOn() DebugOff()).
// Set method must be defined to use this macro.
//
#define BooleanMacro(name,type) \
  virtual void name##On () { this->Set##name(static_cast<type>(1));}   \
  virtual void name##Off () { this->Set##name(static_cast<type>(0));}

//
// Following set macros for vectors define two members for each macro.  The first
// allows setting of individual components (e.g, SetColor(float,float,float)),
// the second allows setting from an array (e.g., SetColor(float* rgb[3])).
// The macros vary in the size of the vector they deal with.
//
#define SetVector2Macro(name,type) \
virtual void Set##name (type _arg1, type _arg2) \
    { \
  if ((this->name[0] != _arg1)||(this->name[1] != _arg2)) \
      { \
    this->name[0] = _arg1; \
    this->name[1] = _arg2; \
    this->Modified(); \
      } \
    } \
void Set##name (type _arg[2]) \
    { \
  this->Set##name (_arg[0], _arg[1]); \
    }

#define GetVector2Macro(name,type) \
virtual type *Get##name () \
{ \
  return this->name; \
} \
virtual void Get##name (type &_arg1, type &_arg2) \
    { \
    _arg1 = this->name[0]; \
    _arg2 = this->name[1]; \
    } \
virtual void Get##name (type _arg[2]) \
    { \
  this->Get##name (_arg[0], _arg[1]);\
    }

#define SetVector3Macro(name,type) \
virtual void Set##name (type _arg1, type _arg2, type _arg3) \
    { \
  if ((this->name[0] != _arg1)||(this->name[1] != _arg2)||(this->name[2] != _arg3)) \
      { \
    this->name[0] = _arg1; \
    this->name[1] = _arg2; \
    this->name[2] = _arg3; \
    this->Modified(); \
      } \
    } \
virtual void Set##name (type _arg[3]) \
    { \
  this->Set##name (_arg[0], _arg[1], _arg[2]);\
    }

#define GetVector3Macro(name,type) \
virtual type *Get##name () \
{ \
  return this->name; \
} \
virtual void Get##name (type &_arg1, type &_arg2, type &_arg3) \
    { \
    _arg1 = this->name[0]; \
    _arg2 = this->name[1]; \
    _arg3 = this->name[2]; \
    } \
virtual void Get##name (type _arg[3]) \
    { \
  this->Get##name (_arg[0], _arg[1], _arg[2]);\
    }

#define SetVector4Macro(name,type) \
virtual void Set##name (type _arg1, type _arg2, type _arg3, type _arg4) \
    { \
  if ((this->name[0] != _arg1)||(this->name[1] != _arg2)||(this->name[2] != _arg3)||(this->name[3] != _arg4)) \
      { \
    this->name[0] = _arg1; \
    this->name[1] = _arg2; \
    this->name[2] = _arg3; \
    this->name[3] = _arg4; \
    this->Modified(); \
      } \
    } \
virtual void Set##name (type _arg[4]) \
    { \
  this->Set##name (_arg[0], _arg[1], _arg[2], _arg[3]);\
    }


#define vtkGetVector4Macro(name,type) \
virtual type *Get##name () \
{ \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer " << this->name); \
  return this->name; \
} \
virtual void Get##name (type &_arg1, type &_arg2, type &_arg3, type &_arg4) \
    { \
    _arg1 = this->name[0]; \
    _arg2 = this->name[1]; \
    _arg3 = this->name[2]; \
    _arg4 = this->name[3]; \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = (" << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << ")"); \
    } \
virtual void Get##name (type _arg[4]) \
    { \
  this->Get##name (_arg[0], _arg[1], _arg[2], _arg[3]);\
    }

#define vtkSetVector6Macro(name,type) \
virtual void Set##name (type _arg1, type _arg2, type _arg3, type _arg4, type _arg5, type _arg6) \
    { \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting " << #name " to (" << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << "," << _arg5 << "," << _arg6 << ")"); \
  if ((this->name[0] != _arg1)||(this->name[1] != _arg2)||(this->name[2] != _arg3)||(this->name[3] != _arg4)||(this->name[4] != _arg5)||(this->name[5] != _arg6)) \
      { \
    this->name[0] = _arg1; \
    this->name[1] = _arg2; \
    this->name[2] = _arg3; \
    this->name[3] = _arg4; \
    this->name[4] = _arg5; \
    this->name[5] = _arg6; \
    this->Modified(); \
      } \
    } \
virtual void Set##name (type _arg[6]) \
    { \
  this->Set##name (_arg[0], _arg[1], _arg[2], _arg[3], _arg[4], _arg[5]);\
    }

#define vtkGetVector6Macro(name,type) \
virtual type *Get##name () \
{ \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer " << this->name); \
  return this->name; \
} \
virtual void Get##name (type &_arg1, type &_arg2, type &_arg3, type &_arg4, type &_arg5, type &_arg6) \
    { \
    _arg1 = this->name[0]; \
    _arg2 = this->name[1]; \
    _arg3 = this->name[2]; \
    _arg4 = this->name[3]; \
    _arg5 = this->name[4]; \
    _arg6 = this->name[5]; \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " = (" << _arg1 << "," << _arg2 << "," << _arg3 << "," << _arg4 << "," << _arg5 <<"," << _arg6 << ")"); \
    } \
virtual void Get##name (type _arg[6]) \
    { \
  this->Get##name (_arg[0], _arg[1], _arg[2], _arg[3], _arg[4], _arg[5]);\
    }

//
// General set vector macro creates a single method that copies specified
// number of values into object.
// Examples: void SetColor(c,3)
//
#define vtkSetVectorMacro(name,type,count) \
virtual void Set##name(type data[]) \
{ \
  int i; \
  for (i=0; i<count; i++) { if ( data[i] != this->name[i] ) { break; }} \
  if ( i < count ) \
      { \
    for (i=0; i<count; i++) { this->name[i] = data[i]; }\
    this->Modified(); \
      } \
}

//
// Get vector macro defines two methods. One returns pointer to type
// (i.e., array of type). This is for efficiency. The second copies data
// into user provided array. This is more object-oriented.
// Examples: float *GetColor() and void GetColor(float c[count]).
//
#define vtkGetVectorMacro(name,type,count) \
virtual type *Get##name () \
{ \
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): returning " << #name " pointer " << this->name); \
  return this->name; \
} \
virtual void Get##name (type data[count]) \
{ \
  for (int i=0; i<count; i++) { data[i] = this->name[i]; }\
}


#endif
