"""Joint Mp/Mw section with the same coverage as the earlier ML>=2 study."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from potency_model import source_vs

R=Path(__file__).resolve().parents[1];T=R/'small_events/extension/tables'
BLUE,RED,ORANGE,GREEN,GRAY='#176da0','#b32438','#bd641f','#298365','#969da1'
Y={'Mw_Kim1983':'Mw','Mw_Kim2011':'Mw_source_Vs_Kim2011','Mp_Kim1983':'Mp_Liu_Kim1983','Mp_Kim2011':'Mp_Liu_Kim2011'}
def read(p):return json.loads(p.read_text())
def csv(p):return pd.read_csv(p,dtype={'event_id':str})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def p(s):return '<p>'+s+'</p>'
def table(head,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'


class PotencySection:
    def __init__(self,save,frame,grid):
        self.save,self.frame,self.grid=save,frame,grid
        self.audit=read(T/'potency_analysis_validation.json');self.controls=read(T/'potency_controls_validation.json')
        assert self.audit['passed'] and self.controls['passed']
        for source,digest in self.audit['sources_sha256'].items():assert sha(R/source)==digest,source
        for proof in [self.audit,self.controls]:
            for name,digest in proof['outputs_sha256'].items():assert sha(T/name)==digest,name
        assert self.controls['analysis_validation_sha256']==sha(T/'potency_analysis_validation.json')
        self.d=csv(T/'potency_event_catalog.csv');self.g=self.d[self.d.paired_A&self.d.ML_Sheen.ge(1.5)]
        self.members=csv(T/'magnitude_scaling_members.csv');self.f=csv(T/'magnitude_scaling_regressions.csv')
        self.m=csv(T/'magnitude_scaling_performance.csv');self.pred=csv(T/'magnitude_scaling_predictions.csv')
        self.bins=csv(T/'magnitude_scaling_bins.csv');self.depth=csv(T/'potency_depth_regressions.csv')
        self.olddepth=csv(R/'tables/magnitude_followup_depth_regressions.csv')
        self.depthboot=np.load(T/'potency_depth_bootstrap.npz');self.boot=np.load(T/'magnitude_scaling_bootstrap.npz')
        self.hong=csv(T/'potency_Hong_matches.csv');self.hs=csv(T/'potency_Hong_statistics.csv')
        self.hf=csv(T/'potency_Hong_regressions.csv');self.summary=read(T/'potency_summary.json')
        self.normal=csv(T/'potency_normalization.csv')
        self.regional=csv(T/'potency_regional_regressions.csv');self.regional_summary=read(T/'potency_regional_summary.json')
        self.ml_controls=csv(T/'potency_ML_definition_controls.csv')

    def fit(self,response='Mp_Kim1983',axis='our',cohort='expanded_NW2',objective='bin_OLS',degree=2):
        q=self.f[self.f.response.eq(response)&self.f.axis.eq(axis)&self.f.cohort.eq(cohort)&self.f.objective.eq(objective)&
            self.f.degree.eq(degree)&self.f.selection.eq('Main methods')];assert len(q)==1
        return q.iloc[0]

    def metric(self,response,degree,subset,objective='bin_OLS'):
        q=self.m[self.m.response.eq(response)&self.m.cohort.eq('expanded_NW2')&self.m.axis.eq('our')&self.m.objective.eq(objective)&
            self.m.degree.eq(degree)&self.m.subset.eq(subset)&self.m.target_set.eq('shared_NW2')&self.m.selection.eq('Main methods')];assert len(q)==1
        return q.iloc[0]

    def dep(self,response,control='ML quadratic',cohort='expanded_NW2'):
        q=self.depth[self.depth.y.eq(Y[response])&self.depth.x.eq('ML_Sheen')&self.depth.control.eq(control)&self.depth.cohort.eq(cohort)];assert len(q)==1
        return q.iloc[0]

    def scaling(self):
        fig,axes=self.frame('Potency, moment and local magnitude on the same events')
        for i,(axis,xkey,xlabel) in enumerate([('our','ML_Sheen','Measured ML'),('KMA','ml','KMA ML')]):
            for j,(response,ylabel) in enumerate([('Mp_Kim1983','Mp'),('Mw_Kim1983','Mw')]):
                ax=axes[i,j];y=Y[response];q=self.fit(response,axis);x=np.linspace(q.x_min,q.x_max,250);u=x-2
                dots=ax.scatter(self.g[xkey],self.g[y],c=self.g.z_central,cmap='viridis',norm=Normalize(5,25),s=10,alpha=.65,ec='none')
                for degree,style in [(1,'--'),(2,'-')]:
                    r=self.fit(response,axis,degree=degree);ax.plot(x,r.a+r.b*u+r.c*u*u,c=BLUE,ls=style,lw=1.4)
                old=self.fit(response,axis,'previous');ax.plot(x,old.a+old.b*u+old.c*u*u,c=ORANGE,ls=':',lw=1.2)
                draws=self.boot[q.case_id];curves=draws[:,0,None]+draws[:,1,None]*u+draws[:,2,None]*u*u
                lo,hi=np.quantile(curves,[.025,.975],axis=0);ax.fill_between(x,lo,hi,color=BLUE,alpha=.10,lw=0)
                if j==0:ax.plot(x,.04*x*x+.556*x+1.133,c=RED,lw=1,label='Liu et al. (2026)')
                ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(1.4,5.5) if i==0 else (.5,5.3),ylim=(1.3,5.4));self.grid(ax)
                ax.set_title(f'{"abcd"[2*i+j]}  {ylabel} versus '+('measured ML' if axis=='our' else 'KMA ML')+f'; c={q.c:+.4f}',loc='left',fontsize=9)
        fig.subplots_adjust(right=.91)
        fig.colorbar(dots,ax=axes,fraction=.021,pad=.025,label='Depth (km)')
        fig.legend([Line2D([],[],c=BLUE,ls='--'),Line2D([],[],c=BLUE),Line2D([],[],c=ORANGE,ls=':'),Line2D([],[],c=RED)],
            ['Bin line','Bin quadratic','Previous bin quadratic','Liu curve (Mp panels)'],loc='lower center',bbox_to_anchor=(.5,.09),ncol=4,frameon=False,fontsize=7.5)
        self.save(fig,'11_extension_Mp_Mw_ML','The same 676 paired-A events and Kim & Kim (1983) source model; Mp uses the original Liu normalization and Mw uses C=9.05.\n'
            'Bin medians have equal fit weight. Shading: 95% whole-family bootstrap. The earlier 357-event curves retain their NW4 values; the Liu curve is unshifted.')

    def methods(self):
        fig,axes=self.frame('Mw and Mp: the same regression methods and uncertainty tests',left=.14)
        objects=['family_OLS','family_ODR','event_OLS','event_ODR','bin_OLS','bin_ODR'];labels=['Family OLS','Family ODR','Event OLS','Event ODR','Bin OLS','Bin ODR']
        for i,response in enumerate(['Mp_Kim1983','Mw_Kim1983']):
            for j,axis in enumerate(['our','KMA']):
                ax=axes[i,j]
                for off,cohort,color,label in [(-.1,'previous',GRAY,'Previous 357'),(.1,'expanded_NW2',BLUE,'Expanded 676')]:
                    for k,obj in enumerate(objects):
                        q=self.fit(response,axis,cohort,obj)
                        ax.errorbar(q.c,k+off,xerr=[[max(0,q.c-q.c_lower)],[max(0,q.c_upper-q.c)]],fmt='o',ms=3,c=color,capsize=2,label=label if k==0 else None)
                ax.axvline(0,c='.5',lw=.8,ls=':');ax.set(yticks=range(6),yticklabels=labels,xlabel='Quadratic coefficient c');ax.invert_yaxis();self.grid(ax)
                ax.set_title(f'{"abcd"[2*i+j]}  {response.split("_")[0]}; '+('measured ML' if axis=='our' else 'KMA ML'),loc='left')
        axes[0,0].legend(frameon=False,fontsize=7,loc='lower right');fig.subplots_adjust(wspace=.58)
        self.save(fig,'12_extension_Mp_Mw_methods','Every row is fitted freely on the same event membership for each cohort and quantity; ODR assumes equal errors on its fitted coordinates.\n'
            'The earlier 128 Mw/Mp specifications and native-target prediction scores reproduce. Tables also retain 10/20/50 bins and the minimum-three-events-per-bin control.')

    def large_events(self):
        fig,axes=self.frame('Large-event fits and predictions: Mp and Mw together')
        big=self.g[self.g.ML_Sheen.gt(4)].sort_values('event_id');x=np.linspace(3.8,5.4,250);u=x-2
        for j,response in enumerate(['Mp_Kim1983','Mw_Kim1983']):
            ax=axes[0,j];r=self.fit(response)
            for degree,style in [(1,'--'),(2,'-')]:
                q=self.fit(response,degree=degree);ax.plot(x,q.a+q.b*u+q.c*u*u,c=BLUE,lw=1.4,ls=style,label='Bin line' if degree==1 else 'Bin quadratic')
            ax.scatter(big.ML_Sheen,big[Y[response]],c=BLUE,s=27,zorder=4)
            for k,row in enumerate(big.itertuples()):ax.annotate(str(k+1),(row.ML_Sheen,getattr(row,Y[response])),xytext=(4,5),textcoords='offset points',fontsize=8)
            ax.set(xlabel='Measured ML',ylabel=response.split('_')[0],xlim=(3.8,5.5),ylim=(3.35,5.2));ax.set_title(f'{"ab"[j]}  Curves fitted to all 676 events',loc='left',fontsize=9);self.grid(ax)
            if j==0:ax.legend(frameon=False,fontsize=7)
            ax=axes[1,j]
            for off,degree,color,label in [(-.18,1,GRAY,'Bin line'),(.18,2,BLUE,'Bin quadratic')]:
                q=self.fit(response,degree=degree);pr=self.pred[self.pred.case_id.eq(q.case_id)&self.pred.target_set.eq('shared_NW2')].set_index('event_id').loc[big.event_id]
                ax.bar(np.arange(6)+off,abs(pr.observed-pr.predicted),.34,color=color,label=label)
            a=self.metric(response,1,'Measured ML>4');b=self.metric(response,2,'Measured ML>4')
            ax.text(.03,.94,f'Event RMS: {a.CV_RMS_events:.3f} → {b.CV_RMS_events:.3f}',transform=ax.transAxes,va='top',fontsize=8)
            ax.set(xticks=np.arange(6),xticklabels=[str(k+1) for k in range(6)],xlabel='Event',ylabel='Absolute held-out prediction error');self.grid(ax)
            ax.set_ylim(0,max(ax.get_ylim()[1]*1.3,.28));ax.set_title(f'{"cd"[j]}  Whole target family omitted',loc='left',fontsize=9)
        axes[1,0].legend(frameon=False,fontsize=7,loc='upper right')
        self.save(fig,'13_extension_Mp_Mw_large_events','Six measured-ML>4 events in five families; all methods use identical NW2 targets. The two large Gyeongju events require extrapolation in their held-out fold.\n'
            'Labels 1–6: 2015-12-21, 2016-09-12, 2016-09-19, 2019-07-21, 2023-11-29 and 2024-06-11 (UTC). Few independent large families limit generalization.')

    def depth_plot(self):
        fig,axes=self.frame('Depth dependence after extending the magnitude range',left=.12);z=np.linspace(4,26,250);zz=(z-15)/10
        for i,quantity in enumerate(['Mp','Mw']):
            for j,control in enumerate(['Raw magnitude difference','ML quadratic']):
                response=quantity+'_Kim1983';row=self.dep(response,control);beta=np.array(json.loads(row.coefficients));ax=axes[i,j]
                if j==0:
                    y=self.g[Y[response]]-self.g.ML_Sheen;line=beta[0]+beta[-1]*zz
                    draws=self.depthboot[row.case_id];curves=draws[:,0,None]+draws[:,-1,None]*zz
                else:
                    xx=self.g.ML_Sheen-3;y=self.g[Y[response]]-(beta[0]+beta[1]*xx+beta[2]*xx*xx);line=beta[-1]*zz
                    curves=self.depthboot[row.case_id][:,-1,None]*zz
                dots=ax.scatter(self.g.z_central,y,c=self.g.ML_Sheen,cmap='viridis',norm=Normalize(1.5,5.5),s=9,alpha=.5,ec='none')
                lo,hi=np.quantile(curves,[.025,.975],axis=0);ax.fill_between(z,lo,hi,color=BLUE,alpha=.12,lw=0);ax.plot(z,line,c=BLUE,lw=1.3,label='Kim & Kim 1983')
                alt=self.dep(quantity+'_Kim2011',control);ab=np.array(json.loads(alt.coefficients))
                ax.plot(z,ab[-1]*zz+(ab[0] if j==0 else 0),c=ORANGE,ls='--',lw=1.2,label='Kim et al. 2011')
                ax.axhline(0,c='.7',lw=.7);ax.axvline(15,c='.65',lw=.7,ls=':');self.grid(ax)
                ax.set(xlabel='Depth (km)',ylabel=quantity+' − ML' if j==0 else quantity+' after ML adjustment',xlim=(4,26))
                ax.set_title(f'{"abcd"[2*i+j]}  '+('Raw difference' if j==0 else 'Joint quadratic ML + depth'),loc='left',fontsize=9)
                ax.text(.03,.96,f'{row.depth_slope_per_10km:+.3f} per 10 km\n[{row.depth_lo:+.3f}, {row.depth_hi:+.3f}]',transform=ax.transAxes,va='top',fontsize=7.7,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1.5))
        axes[0,0].legend(frameon=False,fontsize=7,loc='lower left');fig.subplots_adjust(right=.91)
        fig.colorbar(dots,ax=axes,fraction=.021,pad=.025,label='Measured ML')
        self.save(fig,'14_extension_Mp_Mw_depth','The established family-weighted depth model and 95% whole-family intervals are retained; ML and depth coefficients are estimated jointly.\n'
            'Source-velocity models change only source coefficients. Tables retain geography, sensor/distance, layer, depth-subset and endpoint controls. These are conditional associations.')

    def velocity(self):
        fig,axes=self.frame('Source rigidity, magnitude normalization and depth bounds',left=.12);z=np.linspace(0,27,501)
        ax=axes[0,0]
        for model,color,label in [('Kim1983',BLUE,'Kim & Kim 1983'),('Kim2011',ORANGE,'Kim et al. 2011')]:
            ax.plot(z,2800*source_vs(z,model)**2/1e9,c=color,label=label,lw=1.3)
        ax.set(xlabel='Depth (km)',ylabel='Source rigidity (GPa)');ax.set_title('a  μ(z) = ρ Vs(z)²; ρ = 2800 kg/m³',loc='left');ax.legend(frameon=False,fontsize=7);self.grid(ax)
        ax=axes[0,1]
        for quantity,column,color in [('Mw','Mw_source_Vs_Kim2011',ORANGE),('Mp','Mp_Liu_Kim2011',BLUE)]:
            original='Mw' if quantity=='Mw' else 'Mp_Liu_Kim1983'
            ax.scatter(self.g.z_central,self.g[column]-self.g[original],s=9,c=color,alpha=.45,label=quantity,ec='none')
        ax.axhline(0,c='.6',lw=.7);ax.set(xlabel='Depth (km)',ylabel='Kim2011 − Kim1983 magnitude');ax.set_title('b  Source-coefficient change: ΔMp = ΔMw / 3',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7);self.grid(ax)
        ax=axes[1,0];q=self.normal[self.normal.model.eq('Kim1983')]
        ax.barh(np.arange(4),q.offset_from_Liu,color=[BLUE,GRAY,ORANGE,GREEN]);ax.set(yticks=np.arange(4),yticklabels=q.definition,xlabel='Constant offset from Liu Mp');ax.invert_yaxis();ax.axvline(0,c='.4',lw=.7)
        ax.set_title('c  Reference normalization changes the zero point',loc='left',fontsize=9);self.grid(ax)
        ax=axes[1,1]
        for model,color,label in [('Kim1983',BLUE,'Kim & Kim 1983'),('Kim2011',ORANGE,'Kim et al. 2011')]:
            delta=abs(self.g['Mp_Liu_depth_upper_'+model]-self.g['Mp_Liu_depth_lower_'+model])
            ax.scatter(self.g.upper-self.g.lower,delta,s=10,c=color,alpha=.4,ec='none',label=label)
        ax.set(xlabel='Inherited depth-bound width (km)',ylabel='|Mp(upper) − Mp(lower)|');ax.set_title('d  Source-coefficient endpoint sensitivity',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7);self.grid(ax)
        fig.subplots_adjust(wspace=.58)
        self.save(fig,'15_extension_Mp_velocity_normalization','Source μ uses the event-depth velocity, while the 28/36-GPa values define alternative magnitude zero points. Constant shifts do not change free-fit curvature.\n'
            'ΔMp = ΔMw/3 is algebra for this source-coefficient experiment. Endpoint differences are sensitivities with fixed spectra/path corrections, not depth-posterior errors.')

    def hong_plot(self):
        fig,axes=self.frame('Published Hong magnitudes compared on the same accepted events',left=.12)
        q=self.hong[self.hong.paired_A&self.hong.ML_Sheen.ge(1.5)]
        for j,quantity in enumerate(['Mp','Mw']):
            own=q['Mp_Liu_Kim1983'] if quantity=='Mp' else q.Mw;ax=axes[0,j]
            for label,color,marker in [('spectral',BLUE,'o'),('TD',ORANGE,'^')]:
                x=q['Mp_Hong_'+label] if quantity=='Mp' else q['mw_spectra' if label=='spectral' else 'mw_TD']
                ax.scatter(x,own,s=13,c=color,marker=marker,alpha=.5,ec='none',label='Hong '+label)
            ax.plot([1,5],[1,5],c='.4',ls=':',lw=.8)
            limits=(min(own.min(),q['Mp_Hong_spectral' if quantity=='Mp' else 'mw_spectra'].min())-.15,
                max(own.max(),q['Mp_Hong_TD' if quantity=='Mp' else 'mw_TD'].max())+.15)
            ax.set(xlabel='Hong '+quantity,ylabel='Our '+quantity,xlim=limits,ylim=limits);ax.set_title(f'{"ab"[j]}  {len(q)} paired-A exact event matches',loc='left',fontsize=9);self.grid(ax)
            if j==0:ax.legend(frameon=False,fontsize=7)
            ax=axes[1,j]
            for label,color,marker in [('spectral',BLUE,'o'),('TD',ORANGE,'^')]:
                other=q['Mp_Hong_'+label] if quantity=='Mp' else q['mw_spectra' if label=='spectral' else 'mw_TD']
                ax.scatter(q.z_central,own-other,s=13,c=color,marker=marker,alpha=.5,ec='none')
            ax.axhline(0,c='.4',ls=':',lw=.8);ax.set(xlabel='Source depth (km)',ylabel='Ours − Hong '+quantity);ax.set_title(f'{"cd"[j]}  Differences retain each method’s source constants',loc='left',fontsize=9);self.grid(ax)
        self.save(fig,'16_extension_Mp_Mw_Hong','Exact origin-time matches with both magnitudes grade A. Hong Mp is converted from published spectral/time-domain Mw using Hong’s ρ=2700 kg/m³, Vs=3.61 km/s.\n'
            'No published event magnitude is refitted or shifted to our scale. OLS/ODR tables compare catalog magnitudes and ML; full-catalog regressions retain their different selection.')

    def regional_plot(self):
        fig,axes=self.frame('Local-magnitude definitions and the published Mp curve',left=.13)
        q=self.ml_controls[self.ml_controls.paired_A&self.ml_controls.ML_Sheen.ge(1.5)]
        hb='ML_HB87_same_WA2080';gain='ML_HB87_WA2800_gain_control'
        ax=axes[0,0]
        for deep,color,label in [(False,BLUE,'Depth <15 km'),(True,ORANGE,'Depth ≥15 km')]:
            part=q[q.z_central.ge(15).eq(deep)]
            ax.scatter(part.ML_median_distance,part[hb]-part.ML_Sheen,s=10,c=color,alpha=.45,ec='none',label=label)
        ax.axhline(0,c='.4',lw=.7,ls=':');ax.set(xlabel='Median ML station distance (km)',ylabel='ML attenuation control − measured ML')
        ax.set_title('a  Same accepted WA peaks and stations',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=7);self.grid(ax)
        ax=axes[0,1];x=np.linspace(1,5.5,250)
        for column,color,label in [('ML_Sheen',BLUE,'Sheen, WA2080'),(hb,ORANGE,'HB, WA2080'),(gain,GREEN,'HB, WA2800')]:
            if column=='ML_Sheen':r=self.fit();a,b,c=r.a,r.b,r.c
            else:
                r=self.regional[self.regional.x.eq(column)&self.regional.y.eq('Mp_Liu_Kim1983')&self.regional.method.eq('BinnedOLS')&self.regional.degree.eq(2)].iloc[0]
                a,b,c=r.a_at_ML3-r.slope_at_ML3+r.quadratic_c,r.slope_at_ML3-2*r.quadratic_c,r.quadratic_c
            ax.plot(x,a+b*(x-2)+c*(x-2)**2,c=color,lw=1.3,label=label)
        ax.plot(x,.04*x*x+.556*x+1.133,c=RED,lw=1.1,label='Liu et al. (2026)')
        ax.set(xlabel='Stated ML definition',ylabel='Mp',xlim=(1,5.5),ylim=(1.3,5.4));ax.set_title('b  Free bin-median fits; Mp held fixed',loc='left',fontsize=9);ax.legend(frameon=False,fontsize=6.7);self.grid(ax)
        ax=axes[1,0];labels=[]
        for i,(response,y,method,name) in enumerate([('Mp_Kim1983','Mp_Liu_Kim1983','OLS','Mp family OLS'),('Mp_Kim1983','Mp_Liu_Kim1983','BinnedOLS','Mp bin OLS'),('Mw_Kim1983','Mw','OLS','Mw family OLS'),('Mw_Kim1983','Mw','BinnedOLS','Mw bin OLS')]):
            labels.append(name)
            base=self.fit(response,objective='family_OLS' if method=='OLS' else 'bin_OLS')
            for off,column,color in [(-.15,'ML_Sheen',BLUE),(0,hb,ORANGE),(.15,gain,GREEN)]:
                if column=='ML_Sheen':c,lo,hi=base.c,base.c_lower,base.c_upper
                else:
                    r=self.regional[self.regional.x.eq(column)&self.regional.y.eq(y)&self.regional.method.eq(method)&self.regional.degree.eq(2)].iloc[0]
                    c,lo,hi=r.quadratic_c,r.c_lo,r.c_hi
                ax.errorbar(c,i+off,xerr=[[max(0,c-lo)],[max(0,hi-c)]],fmt='o',c=color,ms=3,capsize=2)
        ax.axvline(0,c='.5',lw=.7,ls=':');ax.set(yticks=range(4),yticklabels=labels,xlabel='Quadratic coefficient c');ax.invert_yaxis();ax.set_title('c  Gain shifts preserve curvature and prediction',loc='left',fontsize=9);self.grid(ax)
        ax=axes[1,1];defs=self.regional_summary['definitions']
        ax.bar(np.arange(4),[d['median_Mp_minus_Liu'] for d in defs],color=[BLUE,GRAY,ORANGE,GREEN])
        ax.axhline(0,c='.4',lw=.7);ax.set(xticks=np.arange(4),xticklabels=['Sheen\n2080','KMA','HB\n2080','HB\n2800'],ylabel='Median Mp − Liu curve')
        ax.set_title('d  Apparent offset depends on ML definition',loc='left',fontsize=9);self.grid(ax)
        fig.subplots_adjust(wspace=.55)
        self.save(fig,'17_extension_ML_definition_control','HB changes only the attenuation/distance convention on the same saved horizontal WA amplitudes; event selection, signal processing and scalar potency stay fixed.\n'
            'The 2800/2080 gain is an illustrative control and shifts ML by 0.1291. This is not a full reproduction of the published Liu observational ML processing.')

    def definition_page(self):
        normal=self.normal[self.normal.model.eq('Kim1983')]
        return ('<h1>Potency, moment and local magnitude</h1>'+p('<b>The extension carries the earlier joint Mw–Mp–ML analysis onto the same frozen sample.</b> Scalar potency and Mp are available for all 2,000 finite Mw cases among 3,532 attempts. The 676 paired-A events in 374 families define the main regression sample. Mp inherits the spectrum’s Mw quality; it is a physical conversion of the same source amplitude, not an independent spectral fit.')+
            '<h2>Same definitions as the earlier ML≥2 report</h2>'+p('<b>P₀ = M₀ / μ(z), μ(z) = ρ Vs(z)²</b>, with P₀ in m³, M₀ in N m and ρ=2800 kg/m³. Source-depth Vs follows Kim & Kim (1983), with Vp/Vs=1.73; Kim et al. (2011) is the comparison. Equivalently, the retained spectral coefficient gives P₀=4πVsΩr/(2×0.63). The direct and moment-based potency calculations are checked independently.')+
            p('<b>Mp = (2/3)[log₁₀(P₀ in m³) + 1.4563]</b>. This is the earlier Liu normalization; the printed mixed-unit equation uses 5.4563 with potency in cm km². Its equivalent reference rigidity is approximately 36 GPa. Source μ(z) remains depth dependent; 36 GPa defines the magnitude zero point.')+
            p('Mw retains C=9.05. The exact Liu Mp corresponds to a 36-GPa-reference Mw-like magnitude at C≈9.10. The alternative C=9.05 normalization is retained explicitly, rather than silently moving the published Liu scale.')+
            table(['Definition','Offset from Liu Mp'],[[r.definition,f'{r.offset_from_Liu:+.6f}'] for r in normal.itertuples()])+
            '<h2>Same methods applied to Mw and Mp</h2>'+p('Every one of the earlier 128 Mw/Mp regression specifications and its native-target prediction score is reproduced. Expanded fits cover measured and KMA ML, both velocity models, linear/quadratic family/event OLS and equal-axis ODR, and equal-weight bin-median OLS. ODR on bin medians is an added sensitivity. The earlier 10/20/50-bin and minimum-three-event bin controls are retained. There are 480 matched magnitude-scaling cases; external and regional control fits are retained separately.')+
            p('The earlier 357-event NW4 curves retain their measured values. Expanded NW4, larger-only NW2 and expanded NW2 fits distinguish spectral-method and sample changes. All shared-target tests predict the same NW2 observations for the stated quantity and model. Entire target families are omitted; bins and weights are recomputed after omission. The earlier attenuation/distance and WA-gain controls are repeated on identical station amplitudes.')+
            p('Regression uncertainty resamples whole families, using 1,000–2,000 draws; actual per-case counts and reused joint draws are retained. The scalar-source conversion gives the same conditional station-bootstrap width for Mw and Mp. Source-depth endpoints are model sensitivities, not posterior samples. Scalar Mp is the quantity estimated here; the effective-path-rigidity magnitude Mpv requires a different waveform-inversion analysis.')+
            p('The all-attempt potency_event_catalog.csv contains ML, Mw, P₀, μ, Mp, normalization alternatives, bounds, quality and failure reasons. Missing Mw gives missing Mp. No magnitude or quality gate is changed to improve a regression.'))

    def coefficient_page(self,axis):
        rows=[]
        for response,label in [('Mp_Kim1983','Mp, K83'),('Mw_Kim1983','Mw, K83'),('Mp_Kim2011','Mp, K11'),('Mw_Kim2011','Mw, K11')]:
            for objective,name in [('bin_OLS','Bin OLS'),('family_OLS','Family OLS')]:
                for degree in [1,2]:
                    q=self.fit(response,axis,objective=objective,degree=degree)
                    rows.append([label,name+(' line' if degree==1 else ' quad'),f'{q.a:.6f}',f'{q.b:.6f}',f'{q.c:+.6f}' if degree==2 else '0',f'[{q.c_lower:+.3f}, {q.c_upper:+.3f}]' if degree==2 else 'fixed'])
        return ('<h1>'+('Measured ML' if axis=='our' else 'KMA ML')+': Mw and Mp equations</h1>'+p('<b>Magnitude = a + bu + cu², u = the stated ML − 2.</b> K83: Kim & Kim (1983), K11: Kim et al. (2011). Every row uses the same 676 events. Six decimal places support reproduction; they are not a claim of magnitude accuracy.')+
            table(['Quantity','Fit','a','b','c','95% interval, c'],rows)+p('Bin fits describe the range of observed magnitudes with equal weight per occupied bin. Family fits assign total weight one per accepted family. Both estimate curvature freely. Whole-family intervals are conditional on the measured catalog and the source/path model.')+
            p('All six OLS/ODR objectives, earlier and expanded cohorts, bin-count/support controls, bootstrap coefficients and held-out predictions are in magnitude_scaling_*. The next figures compare methods, large events, depth and velocity sensitivity. The unchanged published Liu curve is an external reference for Mp.'))

    def interpretation_page(self):
        dep=[]
        for response,label in [('Mp_Kim1983','Mp, K83'),('Mw_Kim1983','Mw, K83'),('Mp_Kim2011','Mp, K11'),('Mw_Kim2011','Mw, K11')]:
            q=self.dep(response);large=self.dep(response,cohort='larger_NW2')
            old=self.olddepth[self.olddepth.y.eq(Y[response])&self.olddepth.x.eq('ML_Sheen')&self.olddepth.control.eq('ML quadratic')].iloc[0]
            dep.append([label,f'{q.depth_slope_per_10km:+.4f}',f'[{q.depth_lo:+.4f}, {q.depth_hi:+.4f}]',f'{large.depth_slope_per_10km:+.4f}',f'{old.depth_slope_per_10km:+.4f}'])
        perf=[]
        for response in ['Mp_Kim1983','Mw_Kim1983']:
            a=self.metric(response,1,'Measured ML>4');b=self.metric(response,2,'Measured ML>4');small=self.metric(response,2,'New measured ML1.5–2')
            perf.append([response.split('_')[0],f'{a.CV_RMS_events:.4f}',f'{b.CV_RMS_events:.4f}',f'{small.CV_RMS_events:.4f}'])
        hs=[]
        for response in ['Mp_Kim1983','Mw_Kim1983']:
            q=self.hs[self.hs.response.eq(response)&self.hs['sample'].eq('Paired A')]
            sp=q[q.Hong_method.eq('spectral')].iloc[0];td=q[q.Hong_method.eq('TD')].iloc[0]
            hs.append([response.split('_')[0],int(sp.n),f'{sp["median"]:+.4f}',f'{td["median"]:+.4f}'])
        regional=self.regional_summary['definitions'];base=regional[0];hb=regional[2]
        return ('<h1>Joint results and their limits</h1>'+'<h2>Large-event prediction with bin medians</h2>'+table(['Quantity','Large: line RMS','Large: quadratic RMS','New small: quadratic RMS'],perf)+
            p('Large targets: six measured-ML>4 events in five families. New small targets: 166 measured-ML1.5–2 events outside the pilot. Scores average events and withhold complete target families. Quadratic bin fits improve the observed large-event prediction comparison for both quantities. Family-bootstrap curvature intervals remain broad, and five families cannot establish a universal upper-end relation.')+
            '<h2>Depth slopes after joint quadratic ML adjustment</h2>'+table(['Quantity','Expanded /10 km','95% family interval','Larger NW2','Earlier NW4'],dep)+
            p('The original 84 depth controls are repeated, with 24 larger-only NW2 controls added. Tables retain geography, sensor/distance, layer intercepts, depth subsets, endpoint and fixed-Vs tests. These slopes are conditional associations. ML calibration, path/site effects and source-model structure can contribute; a residual trend does not identify depth as the cause.')+
            p(f'For this source-coefficient-only comparison, changing Kim1983 to Kim2011 gives median |ΔMw|={self.summary["source_velocity_sensitivity"]["Mw"]["median_abs"]:.4f} and |ΔMp|={self.summary["source_velocity_sensitivity"]["Mp"]["median_abs"]:.4f}. The relation ΔMp=ΔMw/3 is algebra under fixed spectra/geometry. It does not establish absolute model accuracy. Changing a uniform reference rigidity only shifts the intercept.')+
            '<h2>Published Hong spectral and time-domain comparison</h2>'+table(['Quantity','Paired matches','Median ours−spectral','Median ours−TD'],hs)+
            p('Hong Mp is derived from the published Mw with Hong’s own ρ=2700 kg/m³ and Vs=3.61 km/s. The comparison retains each method’s source constants; it does not reprocess Hong waveforms. OLS/ODR tables include our magnitudes versus Hong and Hong versus measured/KMA ML, with the full published catalog kept separate from our selected matches.')+
            p(f'The same-WA-peak attenuation/distance control shifts median ML by {hb["median_ML_minus_ours"]:+.3f}; median Mp minus the Liu curve changes from {base["median_Mp_minus_Liu"]:+.3f} to {hb["median_Mp_minus_Liu"]:+.3f}. A uniform 2800/2080 gain shifts ML by +0.1291 while preserving free-polynomial curvature and prediction errors. This is a controlled definition test, not a new adopted ML scale or a full reproduction of Liu’s processing.')+
            '<h2>Files and scope</h2>'+p('potency_event_catalog.csv retains all attempts. magnitude_scaling_* contains 480 fits and predictions; potency_depth_* contains 108 depth-control cases. potency_Hong_*, potency_normalization.csv and potency_Liu_comparison.csv retain external and normalization controls. The earlier study, measured Mw/ML and spectral quality gates remain unchanged. Mp is scalar source potency magnitude; Mpv is not estimated.'))

    def build(self):
        self.scaling();self.methods();self.large_events();self.depth_plot();self.velocity();self.hong_plot();self.regional_plot()
        return [('text',self.definition_page(),'Potency definitions and matched analysis'),
            ('figure','11_extension_Mp_Mw_ML','Mp and Mw versus both local magnitudes'),
            ('text',self.coefficient_page('our'),'Measured ML: joint magnitude equations'),
            ('text',self.coefficient_page('KMA'),'KMA ML: joint magnitude equations'),
            ('figure','12_extension_Mp_Mw_methods','Mp and Mw: regression methods'),
            ('figure','13_extension_Mp_Mw_large_events','Mp and Mw: large-event fits and prediction'),
            ('figure','14_extension_Mp_Mw_depth','Mp and Mw: depth dependence'),
            ('figure','15_extension_Mp_velocity_normalization','Rigidity, normalization and depth bounds'),
            ('figure','16_extension_Mp_Mw_Hong','Published Hong comparison'),
            ('figure','17_extension_ML_definition_control','Regional ML and WA-gain controls'),
            ('text',self.interpretation_page(),'Joint magnitude findings and scope')]
