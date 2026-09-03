; ModuleID = 'sample.c'
source_filename = "sample.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @sum_loop(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s.next, %loop ]
  %t = mul i32 %i, 2
  %h = add i32 %t, 1
  %s.next = add i32 %s, %h
  %i.next = add i32 %i, 1
  %done = icmp sge i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i32 [ 0, %entry ], [ %s.next, %loop ]
  ret i32 %result
}

define i32 @dot(ptr %a, ptr %b, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %s = phi i32 [ 0, %entry ], [ %s.next, %loop ]
  %idx = sext i32 %i to i64
  %pa = getelementptr i32, ptr %a, i64 %idx
  %va = load i32, ptr %pa
  %pb = getelementptr i32, ptr %b, i64 %idx
  %vb = load i32, ptr %pb
  %prod = mul i32 %va, %vb
  %s.next = add i32 %s, %prod
  %i.next = add i32 %i, 1
  %done = icmp sge i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %result = phi i32 [ 0, %entry ], [ %s.next, %loop ]
  ret i32 %result
}
