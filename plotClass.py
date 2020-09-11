# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 23:34:54 2020

@author: vpadu
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

class drawPlot:
    def __init__(self,plt):
        self.fig, self.ax = plt.subplots()
        #fig.clf()
        return
        
    def draw(self,plt,y0,y1,epoche, yLabel,legend_y0,legend_y1,title):
        tick_spacing = 10
        self.ax.plot(epoche,y0)
        self.ax.plot(epoche,y1)
        self.ax.legend([legend_y0, legend_y1])
        self.ax.set(xlabel='Epoche (#)', ylabel=yLabel,
               title=title)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.show()
        
    def clear(self,plt):
        plt.close()
        
#    def draw(self,plt,acc_train,loss_train,acc_val,loss_val,epoche):
#        tick_spacing = 2.5
#        self.ax.plot(epoche,acc_train)
#        self.ax.plot(epoche,loss_train)
#        self.ax.plot(epoche,acc_val)
#        self.ax.plot(epoche,loss_val)
#        self.ax.legend(["AccTrain", "LossTrain","AccVal", "LossVal"])
#        self.ax.set(xlabel='Epoche (#)', ylabel='Accurancy-Loss (%)',
#               title='Accurancy/Loss Values')
#        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#        plt.show()
#        plt.plot(epoche,acc_train)
#        plt.plot(epoche,loss_train)
#        plt.plot(epoche,acc_val)
#        plt.plot(epoche,loss_val)
        
        
#        p1, = self.host.plot(epoche,acc_train, label="Acc_train")
#        p2, = self.par1.plot(epoche,loss_train, label="Loss_train")
#        p3, = self.par2.plot(epoche,acc_val, label="Acc_Val")
#        p3, = self.par3.plot(epoche,loss_val, label="Loss_Val")
#        self.par1.set_ylim(0,5)
#        self.par2.set_ylim(0,5)
#        self.host.legend()
#        self.host.axis["left"].label.set_color(p1.get_color())
#        self.par1.axis["right"].label.set_color(p2.get_color())
#        self.par2.axis["right"].label.set_color(p3.get_color())
#        self.par3.axis["right"].label.set_color(p3.get_color())
#        plt.draw()
#        plt.show()
        
        
#acc_train,Loss_train,Acc_Val,Loss_Val,epoche = [0, 1, 2],[0, 4, 4],[2, 5, 3],[3, 4, 5],[0, 1, 3]
#plot=drawPlot(plt)
#plot.draw(plt,acc_train,Loss_train,Acc_Val,Loss_Val,epoche)
#plt.close()
        