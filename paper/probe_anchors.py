# -*- coding: utf-8 -*-
import io

P3 = '/home/zhaosiying/codebase/fp4vla/paper/sections/03-方法.md'
P4 = '/home/zhaosiying/codebase/fp4vla/paper/sections/04-实验.md'
P14 = '/home/zhaosiying/codebase/fp4vla/paper/sections/14-附录A-核心源码走读与APXInf框架解析.md'

t3 = io.open(P3, encoding='utf-8').read()
i = t3.find('收敛后才进入闭环实验')
print('A1 tail:', repr(t3[i:i+14]))
t4 = io.open(P4, encoding='utf-8').read()
print('A2 has e1 fig line:', '{{fig:e1_latency}}' in t4)
i = t4.find('小投影进 FP4 只有开销没有收益')
print('A3 tail:', repr(t4[i:i+14]))
i = t4.find('| GeGLU 4304')
print('A4 row:', repr(t4[i:i+38]))
t14 = io.open(P14, encoding='utf-8').read()
i = t14.find('CudaBuffer` 上传')
print('A5 tail:', repr(t14[i:i+56]))
