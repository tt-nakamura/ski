import numpy as np
import matplotlib.pyplot as plt
from schanze import X,Y,h

x = np.linspace(0, X[-1], 100)
r = -Y[-1]/X[-1]

plt.figure(figsize=(5, 5*r))
plt.axis('equal')
plt.plot(x, h(x), 'k')
plt.plot([0], [0], 'k*')
plt.box('off')
plt.xticks(())
plt.yticks(())
#plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
