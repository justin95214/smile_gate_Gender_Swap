from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt

a ="./GAN/runs/Oct12_12-02-34_isys313/events.out.tfevents.1665543754.isys313.2464797.0"
event_acc = EventAccumulator(a)
event_acc.Reload()

loss_G_GAN = np.array([[s.step, s.value] for s in event_acc.Scalars('loss_G_GAN')])
loss_G_recon = np.array([[s.step, s.value] for s in event_acc.Scalars('loss_G_recon')])
loss_D =np.array([[s.step, s.value] for s in event_acc.Scalars('loss_D')])

plt.subplot(131)
plt.title("loss_G_GAN")
plt.plot(loss_G_GAN[:, 0], loss_G_GAN[:, 1])
plt.subplot(132)
plt.title("loss_G_recon")
plt.plot(loss_G_recon[:, 0], loss_G_recon[:, 1])
plt.subplot(133)
plt.title("loss_D")
plt.plot(loss_D[:, 0], loss_D[:, 1])
plt.show()
