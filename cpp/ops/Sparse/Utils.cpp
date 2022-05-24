/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"
#include "JavaClassStr.h"

#include <map>
#include <algorithm>
#include <stdint.h>
#include <vector>

namespace ops {

static const std::string OOM_ERROR_FQ_NAME = "java/lang/OutOfMemoryError";

// Throw a Java OutOfMemoryError
void out_of_memory(JNIEnv *env) {
  jclass oomErrorClass = env->FindClass(OOM_ERROR_FQ_NAME.c_str());
  // We should not fail to find the OOM Error class, but if by chance we do,
  // raise the error raised from that.
  if (oomErrorClass == NULL)
    return;
  env->ThrowNew(oomErrorClass, "");
}

/** Given the short class name find the Java class */
jclass findClass(JNIEnv * env, const std::string & classname)
{
  jclass c = env->FindClass(classname.c_str());
  Require(c != JNI_FALSE, std::string("Unable to find class ") + classname);
  return c;
}

/** Return an object field from an object
 *
 * This returns the object that is inside of another given object.  It is
 * referenced by the name, the signature, and the class name.
 */
jobject getJObjectFromClass(JNIEnv *env, jobject obj,
    const char * name,
    const char * objsignature,
    const char * classname) {
  jclass clazz = findClass(env, classname);
  jfieldID fieldID = (env)->GetFieldID(clazz, name, objsignature);
  jobject jobj = (env)->GetObjectField(obj, fieldID);
  return jobj;
}

/** Return an int field from an object */
INT getIntFromClass(JNIEnv *env, jobject obj,
    const char *name,
    const char *classname = (char *)J_SparseEigen2D) {
  jclass clazz = findClass(env, classname);
  jfieldID fieldID = (env)->GetFieldID(clazz, name, "I");
  jint i = env->GetIntField(obj, fieldID);
  return (INT)i;
}

/** Return a vector from a Java array
 *
 * @tparam D the C++ data type that needed
 * @tparam T the C++ data type mapping to JavaT
 * @tparam JavaT the Java array type
 */
template<typename D, typename T, typename JavaT>
Array<D> getPrimitiveArray(JNIEnv *env, JavaT data)
{
  size_t size = env->GetArrayLength(data);
  T *cdata = (T *)env->GetPrimitiveArrayCritical(data, 0);
  if (cdata == nullptr) {
    out_of_memory(env);
    // Return dummy result. Caller is responsible for checking if an exception
    // occurred
    return {};
  }
  // copy the memory from cdata to acdata
  Array<D> acdata(cdata, size);
  env->ReleasePrimitiveArrayCritical(data, cdata, 0);
  return acdata;
}

/** Return an int vector from a Java object */
Array<INT> getIntArrayFromClass(JNIEnv *env, jobject obj, const char *name,
    const char *classname = (char *)J_SparseEigen2D) {
  jobject mvdata = getJObjectFromClass(env, obj, name, "[I", classname);
  jintArray *arr = reinterpret_cast<jintArray *>(&mvdata);
  return getPrimitiveArray<INT, int32_t, jintArray>(env, *arr);
}

/** Return a float vector from a Java object */
Array<FLOAT> getFloatArrayFromClass(JNIEnv *env, jobject obj, const char *name,
    const char *classname = (char *)J_SparseEigen2D) {
  jobject mvdata = getJObjectFromClass(env, obj, name, "[F", classname);
  jfloatArray *arr = reinterpret_cast<jfloatArray *>(&mvdata);
  return getPrimitiveArray<FLOAT, float, jfloatArray>(env, *arr);
}

/** Return an int vector from a Java tensor shape field */
Array<INT> javaToShape(JNIEnv * env, jobject tensor) {
   jobject jshape = getJObjectFromClass(env, tensor, "shape",
       J_SIG(J_Shape), J_SparseFloatTensor);
   return getIntArrayFromClass(env, jshape, (char *)"dims", J_Shape);
}

/** Return the method ID with the given function name, class name, and
 * signature */
jmethodID getMethodID(JNIEnv * env, const std::string functionname,
    const std::string signature, const std::string classname)
{
  jmethodID jm = env -> GetMethodID(
      findClass(env, classname.c_str()), functionname.c_str(), signature.c_str());
  Require(jm != JNI_FALSE, std::string("Unable to find the ") + functionname + " with " + signature + " method for " + classname);

  return jm;
}

/** Return the method ID with the given function name and class name */
jmethodID getMethod(JNIEnv * env, const std::string functionname, const std::string classname = J_List)
{
  std::map<std::pair<std::string, std::string>, std::string> nameSign;
  nameSign.insert({std::make_pair(J_List, "size"), "()I"});
  nameSign.insert({std::make_pair(J_List, "get"), "(I)" J_SIG(J_Object)});
  nameSign.insert({std::make_pair(J_ArrayList, "add"), "(" J_SIG(J_Object) ")Z"});
  nameSign.insert({std::make_pair(J_ArrayList, "<init>"), "(I)V"});
  nameSign.insert({std::make_pair(J_SparseFloatTensor, "<init>"), "(" J_SIG(J_Shape) "[F" J_SIG(J_List) ")V"});

  auto it = nameSign.find(std::make_pair(classname, functionname));
  Require(it != nameSign.end(), std::string("Unknown signature for the ") + functionname + " method for " + classname);

  return getMethodID(env, functionname, it->second, classname);
}

/** Return DimData from a Java DimData object */
DimData javaDimDataToDimData(JNIEnv * env, jobject jdim) {
  auto inner = getIntArrayFromClass(env, jdim, (char *)"inner", J_DimData);
  auto outer = getIntArrayFromClass(env, jdim, (char *)"outer", J_DimData);
  return {std::move(inner), std::move(outer)};
}

/** Return a DimData vector from a Java sparse float tensor object */
std::vector<DimData> javaToDimDataVector(JNIEnv * env, jobject tensor) {
   std::vector<DimData> dims;

   // find the list of DimData
   jobject jdims = getJObjectFromClass(env, tensor, "dims", J_SIG(J_List), J_SparseFloatTensor);

   // find the methods to iterate the list
   jmethodID mSize = getMethod(env, "size");
   jmethodID mGet = getMethod(env, "get");

   // get the size of the list
   jint size = env->CallIntMethod(jdims, mSize);

   dims.reserve(size);

   for(jint i=0; i<size; i++) {
     jobject jdim = env->CallObjectMethod(jdims, mGet, i);
     dims.push_back(javaDimDataToDimData(env, jdim));
   }

   return dims;
}

SparseFloatTensor javaToCPPSparseTensor(JNIEnv * env, jobject tensor) {
  Array<INT> shape = javaToShape(env, tensor);
  Array<FLOAT> values = getFloatArrayFromClass(env, tensor, (char *)"values", J_SparseFloatTensor);
  std::vector<DimData> dims = javaToDimDataVector(env, tensor);
  SparseFloatTensor t(shape, values, dims);

  #ifdef DEBUG
  t.checkShapeAndDim();
  #endif // DEBUG

  return t;
}

/** Return a Java int array from a vector of int arrays */
jintArray copyCPPArrayToJava(JNIEnv *env, const Array<INT> & iArray)
{
  jintArray i = (env)->NewIntArray(iArray.size());
  if (typeid(INT) == typeid(int32_t)) // direct set the region
    (env)->SetIntArrayRegion(i, 0, iArray.size(), (int32_t *)iArray.data());
  else { // convert the data to int32_t and then set the region
    Array<int32_t> iArray_(iArray.data(), iArray.size());
    (env)->SetIntArrayRegion(i, 0, iArray_.size(), iArray_.data());
  }
  return i;
}

/** Return a Java float array from a vector of float arrays */
jfloatArray copyCPPArrayToJava(JNIEnv *env, const Array<FLOAT> & fArray)
{
  jfloatArray f = (env)->NewFloatArray(fArray.size());
  if ((std::is_same<FLOAT, float>::value)) // direct set the region
    (env)->SetFloatArrayRegion(f, 0, fArray.size(), (const float *)fArray.data());
  else { // copy the data and then convert to float
    Array<float> fArray_(fArray.data(), fArray.size());
    (env)->SetFloatArrayRegion(f, 0, fArray_.size(), fArray_.data());
  }
  return f;
}

/** Copy a vector of int to a Java Shape object */
jobject copyShapeToJava(JNIEnv * env, const Array<INT> & shape)
{
  jintArray jshapedims = copyCPPArrayToJava(env, shape);

  jclass clazz = findClass(env, J_Shape);
  jmethodID constructor = (env)->GetMethodID(clazz, "<init>", "([I)V");
  jobject obj = (env)->NewObject(clazz, constructor, jshapedims);
  return obj;
}

/** Copy a C++ DimData to a Java DimData */
jobject copyDimDataToJava(JNIEnv * env, const DimData & dim)
{
  jintArray jinner = copyCPPArrayToJava(env, dim.inner());
  jintArray jouter = copyCPPArrayToJava(env, dim.outer());

  jclass clazz = findClass(env, J_DimData);
  // Get the method id of an empty constructor in clazz
  jmethodID constructor = (env)->GetMethodID(clazz, "<init>", "([I[I)V");
  // Create an instance of clazz
  jobject obj = (env)->NewObject(clazz, constructor, jinner, jouter);
  return obj;
}

/** Copy a vector of C++ DimData to a Java ArrayList object */
jobject copyDimDataVectorToJava(JNIEnv * env, const std::vector<DimData> & dims) {

  jclass clazz = findClass(env, J_ArrayList);
  jmethodID constructor = getMethod(env, "<init>", J_ArrayList);

  jobject jdims = (env)->NewObject(clazz, constructor, dims.size());

  jmethodID madd = getMethod(env, "add", J_ArrayList);

  // iterate through the dims
  for (size_t i=0; i<dims.size(); i++) {
    jobject jdim = copyDimDataToJava(env, dims[i]);
    env->CallVoidMethod(jdims, madd, jdim);
  }

  return jdims;
}

jobject cppToJavaSparseTensor(JNIEnv *env, const SparseFloatTensor & tensor) {
  jobject jshape = copyShapeToJava(env, tensor.shape());
  jfloatArray jvalues = copyCPPArrayToJava(env, tensor.values());
  jobject jdims = copyDimDataVectorToJava(env, tensor.dims());

  jclass clazz = findClass(env, J_SparseFloatTensor);
  jmethodID constructor = getMethod(env, "<init>", J_SparseFloatTensor);
  jobject obj = env -> NewObject(clazz, constructor, jshape, jvalues, jdims);

  return obj;
}

COO javaToCOO(JNIEnv *env, jintArray shape,
    jintArray rows,
    jintArray cols,
    jfloatArray values) {

  Array<INT> shapeData = getPrimitiveArray<INT, int32_t, jintArray>(env, shape);
  INT r = shapeData[0]; INT c = shapeData[1];
  Array<INT> rData = getPrimitiveArray<INT, int32_t, jintArray>(env, rows);
  Array<INT> cData = getPrimitiveArray<INT, int32_t, jintArray>(env, cols);
  Array<FLOAT> vData = getPrimitiveArray<FLOAT, float, jfloatArray>(env, values);

  return COO(r, c, rData, cData, vData);
}


} // namespace ops
