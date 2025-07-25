; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-nvidia-cuda"

define <6 x half> @half6() {
; CHECK-LABEL: half6(
; CHECK:       {
; CHECK-EMPTY:
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    st.param.b32 [func_retval0+8], 0;
; CHECK-NEXT:    st.param.v2.b32 [func_retval0], {0, 0};
; CHECK-NEXT:    ret;
  ret <6 x half> zeroinitializer
}

define <10 x half> @half10() {
; CHECK-LABEL: half10(
; CHECK:       {
; CHECK-EMPTY:
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    st.param.b32 [func_retval0+16], 0;
; CHECK-NEXT:    st.param.v2.b32 [func_retval0+8], {0, 0};
; CHECK-NEXT:    st.param.v2.b32 [func_retval0], {0, 0};
; CHECK-NEXT:    ret;
  ret <10 x half> zeroinitializer
}

define <12 x i8> @byte12() {
; CHECK-LABEL: byte12(
; CHECK:       {
; CHECK-EMPTY:
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    st.param.b32 [func_retval0+8], 0;
; CHECK-NEXT:    st.param.b64 [func_retval0], 0;
; CHECK-NEXT:    ret;
  ret <12 x i8> zeroinitializer
}

define <20 x i8> @byte20() {
; CHECK-LABEL: byte20(
; CHECK:       {
; CHECK-EMPTY:
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    st.param.b32 [func_retval0+16], 0;
; CHECK-NEXT:    st.param.b64 [func_retval0+8], 0;
; CHECK-NEXT:    st.param.b64 [func_retval0], 0;
; CHECK-NEXT:    ret;
  ret <20 x i8> zeroinitializer
}
