#模型训练后的参数可视化
import matplotlib.pyplot as plt
import xlrd
from matplotlib import ticker
import numpy as np

fig, ax = plt.subplots()

data=xlrd.open_workbook(r'E:\UTB-finish\UTB_master\second.xls')
table=data.sheets()[0]
print(table.nrows)
print(table.ncols )
print(table.row_values(0))

#创建每列为一个列表，然后分别从每一列读数据放在列表中
iter=[]
loss_ce=[]
loss_fm=[]
loss_S=[]
loss_D=[]
loss_st=[]

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,0)
    iter.append(cell_value)

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,1)
    loss_ce.append(cell_value)

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,2)
    loss_fm.append(cell_value)

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,3)
    loss_S.append(cell_value)

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,4)
    loss_D.append(cell_value)

for row_index in range(table.nrows):
    cell_value=table.cell_value(row_index,5)
    loss_st.append(cell_value)

plt.subplot(2,3,1)   #loss_ce
plt.plot(iter,loss_ce)
plt.title('loss_ce')
plt.xlabel('iter')     #x y轴名字
plt.ylabel('loss_ce')
ax.set_xlim(left=0)
# plt.yticks(fontsize=5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

plt.subplot(2,3,2)
plt.plot(iter,loss_fm)
plt.title('loss_fm')
plt.xlabel('iter')
plt.ylabel('loss_fm')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))

plt.subplot(2,3,3)
plt.plot(iter,loss_S)
plt.title('loss_S')
plt.xlabel('iter')
plt.ylabel('loss_S')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))

plt.subplot(2,3,4)
plt.plot(iter,loss_D)
plt.title('loss_D')
plt.xlabel('iter')
plt.ylabel('loss_D')
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))

plt.subplot(2,3,5)
plt.plot(iter,loss_st)
plt.title('loss_st')
plt.xlabel('iter')
plt.ylabel('loss_st')
plt.ylim(ymin=0, ymax= 1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.show()

