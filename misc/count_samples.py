Xs = [300,400,500,600,700,800,900,1000]
Ys = [50,70,80,90,100,125,150,200,250,300,400,500,600,700,800]
#N_Ys = [6,8,9,10,11,12,13,14]
#Ys = [100,125,150,200,250,300,400,500,600,700,800]
#Ys = [70, 80, 90, 100, 125]
#N_Ys = [5,5,5,5,5,5,5,5]

N_Ys_check = []
for X in Xs:
  N_Ys_check.append(0)
  for i, Y in enumerate(Ys):
    if X-Y > 125:
      N_Ys_check[-1] += 1

#assert N_Ys == N_Ys_check, print(N_Ys, N_Ys_check)
N_Ys = N_Ys_check

points = []
for i in range(len(Xs)):
    for j in range(N_Ys[i]):
        points.append((Xs[i],Ys[j]))

print(len(points))