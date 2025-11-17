library(tableone)
library(lme4)
library(lmerTest)
library(survival)
library(gtools)
library(survminer)
library(cmprsk)
library(visdat)
library(ggpubr)
library(adjustedCurves)
library(kableExtra)
library(visdat)
library(broom.mixed)

set.seed(987)

EFRAIM<-EFRAIM3_VANCE_VD
head(EFRAIM)
tail(EFRAIM[1:9854,])
EFRAIM<-EFRAIM[1:9854,]
str(EFRAIM)
names(EFRAIM)


EFRAIM$age<-as.numeric(as.character(EFRAIM$Age))
EFRAIM$GENDER<-factor(EFRAIM$Gender,levels=c(0,1), labels=c("Male","Female"))
EFRAIM$Hematological_Malignancy<-factor(EFRAIM$Hem_mal,levels=c(0:9),labels=c("None","AML","ALL","NHL","Myeloma","Hodgkin","CLL","CML","MDS","Other"))
EFRAIM$Hematological_Malignancy[is.na(EFRAIM$Hematological_Malignancy)]<-"None"
EFRAIM$HM_Status<-factor(EFRAIM$Status_HemMal,levels=c(0:6),labels=c("Unknown","Diagnosis","First_Line","Second_line+","Remission","Uncontrolled","Palliative"))
EFRAIM$HSCT<-factor(EFRAIM$HSCT_BMT,levels=c(0:2),labels=c("None","Autologous","Allogeneic"))
EFRAIM$Systemic_Disease<-factor(EFRAIM$Sys_dis,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Solid_tumor<-factor(EFRAIM$Solid_tumor,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Solid_Organ_Transplant<-factor(EFRAIM$Organ_transpl,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Solid_Organ_Transplant[is.na(EFRAIM$Solid_Organ_Transplant)]<-"No"
EFRAIM$Immunosuppr_drugs<-factor(EFRAIM$Drug_induced,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Immunosuppr_drugs[is.na(EFRAIM$Immunosuppr_drugs)]<-"No"
EFRAIM$Chemotherapy_Previous<-factor(EFRAIM$Chemotherapy,levels=c(0,1), labels=c("No","Yes"))
EFRAIM$Ibru_Fluda_MTX_Previous<-factor(EFRAIM$Ibr_Flu_Met,levels=c(0,1), labels=c("No","Yes"))
EFRAIM$CART_Previous<-factor(EFRAIM$CART,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Steroids_Previous<-factor(EFRAIM$Steroids_YN,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$BiTE_Previous<-factor(EFRAIM$BiTE,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Immuno_Drugs_Previous<-factor(EFRAIM$Immuno_drugs,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Targeted_therapy_Previous<-factor(EFRAIM$Tar_ther,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Immunotherapy_Previous<-factor(EFRAIM$Immuno_drugs,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$ECOG<-as.factor(EFRAIM$ECOG)
EFRAIM$Prophylaxis_antifungal<-factor(EFRAIM$Prophylaxis_antifungal,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Prophylaxis_pneumocystis<-factor(EFRAIM$Prophylaxis_pneumocystis,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Prophylaxis_bacterial<-factor(EFRAIM$Prophylaxis_bacterial,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Prophylaxis_viral<-factor(EFRAIM$Prophylaxis_viral,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Goals_of_care[is.na(EFRAIM$Goals_of_care)]<-4
EFRAIM$Goals_of_care<-factor(EFRAIM$Goals_of_care,levels=c(0:5),labels=c("Full_Code","Time_Limited","DNI","DNR","Unknown","Palliative"))
EFRAIM$Full_Code<-as.factor(ifelse(EFRAIM$Goals_of_care=="Full_Code","Yes","No"))
EFRAIM$Temperature<-as.numeric(EFRAIM$Temp/10)
EFRAIM$Leukocytes<-as.numeric(as.character(Leuko$LeukoVD))/10
EFRAIM$Neutrophils<-as.numeric(as.character(Leuko$NeutropVD))/10
EFRAIM$Platelets_VD<-as.numeric(as.character(EFRAIM$Platelets_VD))
EFRAIM$Location_before_ICU<-factor(EFRAIM$Location_before_ICU,levels=c(0:4),labels=c("ED","Ward","Other ICU","Other Hospital","Other"))

EFRAIM$Hospital_Mortality<-factor(EFRAIM$`hospital death`,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$ICU_Mortality<-factor(EFRAIM$`icu death`,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Day90_Mortality<-factor(EFRAIM$`D90 DEATH`,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Comp_risk_IOT<-as.factor(ifelse(EFRAIM$IMV_applicable_YN...121==1,"iMV",
                                       ifelse(EFRAIM$`icu death`==0,"Discharge","Death")))
EFRAIM$Comp_risk_IOT[is.na(EFRAIM$Comp_risk_IOT)]<-ifelse(EFRAIM$`icu death`==0,"Discharge","Death")

EFRAIM$`HOSPITAL   LOS`[is.na(EFRAIM$`HOSPITAL   LOS`)]<-EFRAIM$`ICU LOS`[is.na(EFRAIM$`HOSPITAL   LOS`)]
EFRAIM$`HOSPITAL   LOS`[is.na(EFRAIM$`HOSPITAL   LOS`)]<-21
EFRAIM$Follow_up_Comp<-0
a<-which(EFRAIM$Comp_risk_IOT=="iMV")
b<-which(EFRAIM$Comp_risk_IOT!="iMV")
EFRAIM$Follow_up_Comp[a]<-EFRAIM$"J IOT...122"[a]
EFRAIM$Follow_up_Comp[b]<-EFRAIM$`HOSPITAL   LOS`[b]

EFRAIM$Initial_Ox_Strategy[EFRAIM$Initial_Ox_Strategy=="None_reported"]<-"Standard_O2"
EFRAIM$Initial_Ox_Strategy[EFRAIM$Initial_Ox_Strategy=="NIV"&!is.na(EFRAIM$"HFNO_HFNO_D1 - FiO2")]<-"NIV&HFNO"
EFRAIM$Initial_Oxygenation_Strategy<-as.factor(EFRAIM$Initial_Ox_Strategy)



EFRAIM$Initial_Oxygenation_Strategy_2<-as.factor(ifelse(EFRAIM$Initial_Oxygenation_Strategy=="Standard_O2","Standard_O2", 
                                                        ifelse(EFRAIM$Initial_Oxygenation_Strategy=="CPAP"|EFRAIM$Initial_Oxygenation_Strategy=="NIV"|EFRAIM$Initial_Oxygenation_Strategy=="NIV&HFNO","CPAP/NIV",
                                                               ifelse(EFRAIM$Initial_Oxygenation_Strategy=="iMV","iMV","HFNO"))))

EFRAIM$Diag_Simpl<-as.factor(EFRAIM$Diag_Simpl)
EFRAIM$Diag_simpl
EFRAIM$Diag_Complex<-EFRAIM$Diag_Simpl
EFRAIM$Diag_Complex<-as.character(EFRAIM$Diag_Complex)
EFRAIM$Diag_Complex[EFRAIM$DG1=="EMPTY EMPTY"]<-"No_reported_diagnostic_Work_up"
EFRAIM$Diag_Complex<-as.factor(EFRAIM$Diag_Complex)


EFRAIM$Neutropenia<-"No"
EFRAIM$Neutropenia[EFRAIM$Neutrophils<0.5]<-"Yes"

EFRAIM$ARDS_Berlin<-as.factor(ifelse(EFRAIM$ARDS_ICU_spec==2,"Severe",
                           ifelse(EFRAIM$ARDS_ICU_spec==1,"Moderate",
                                  ifelse(EFRAIM$ARDS_ICU_spec==0,"Mild","NoARDS"))))

EFRAIM$Vasopressors[is.na(EFRAIM$Vasopressors)]<-0
EFRAIM$Vasopressors<-factor(EFRAIM$Vasopressors,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Coma_CNS[is.na(EFRAIM$Coma_CNS)]<-0
EFRAIM$Coma_CNS<-factor(EFRAIM$Coma_CNS,levels=c(0,1),labels=c("No","Yes"))
EFRAIM$Chemotherpy_ICU[is.na(EFRAIM$Chemotherpy_ICU)]<-0
EFRAIM$Chemotherapy_ICU<-EFRAIM$Chemotherpy_ICU

EFRAIM$Diagnostic_Work_Up<-Diag_WorkUp$Diagnostic_Work_Up
EFRAIM$Diagnostic_Work_Up<-as.factor(EFRAIM$Diagnostic_Work_Up)


EFRAIM$Antibiotic_ICU_adequate[is.na(EFRAIM$Antibiotic_ICU_adequate)]<-4
EFRAIM$Antibiotic_ICU_adequate<-factor(EFRAIM$Antibiotic_ICU_adequate,levels=c(0:4),labels=c("No","Yes","Yes_delayed","Never","Unknown"))

a<-which(EFRAIM$`HOSPITAL   LOS`>90&EFRAIM$`D90 DEATH`==1)
b<-which(EFRAIM$`HOSPITAL   LOS`>90&EFRAIM$`D90 DEATH`==0)
EFRAIM$`ICU LOS`[a]
EFRAIM$Hospital_Mortality[a]
EFRAIM$Hospital_LOS<-EFRAIM$`HOSPITAL   LOS`
EFRAIM$Hospital_LOS[a]<-EFRAIM$`ICU LOS`[a]
EFRAIM$Hospital_LOS[b]<-NA


EFRAIM$Follow_up_Comp<-0
a<-which(EFRAIM$Comp_risk_IOT=="iMV")
b<-which(EFRAIM$Comp_risk_IOT!="iMV")
EFRAIM$Follow_up_Comp[a]<-EFRAIM$"J IOT...122"[a]
EFRAIM$Follow_up_Comp[b]<-EFRAIM$Hospital_LOS[b]


EFRAIM$Underlying_ImmuneDefect<-as.factor(ifelse(EFRAIM$Hematological_Malignancy!="None"|EFRAIM$HSCT!="None","Hematological_Malignancy",
                                                 ifelse(EFRAIM$Solid_Organ_Transplant=="Yes","SOT",
                                                       ifelse(EFRAIM$Solid_tumor=="Yes","Solid_Tumor","Systemic_disease_or_ID_drugs") )))
EFRAIM$Underlying_ImmuneDefect[is.na(EFRAIM$Underlying_ImmuneDefect)]<-"Systemic_disease_or_ID_drugs"


EFRAIM$DiagVD<-as.factor(ifelse(EFRAIM$Diag_Complex=="BACTERIAL"|EFRAIM$Diag_Complex=="EXTRAPULMONARY"|EFRAIM$Diag_Complex=="BK"|EFRAIM$Diag_Complex=="ASPIRATION","Bacterial",
                                                 ifelse(EFRAIM$Diag_Complex=="CPO"|EFRAIM$Diag_Complex=="EMBOLISM","Cardiac",
                                                        ifelse(EFRAIM$Diag_Complex=="FUNGAL"|EFRAIM$Diag_Complex=="PJP","Fungal",
                                                               ifelse(EFRAIM$Diag_Complex=="VIRAL","Viral",
                                                                      ifelse(EFRAIM$Diag_Complex=="INFILTRATION"|EFRAIM$Diag_Complex=="Toxicity","Specific",
                                                                             ifelse(EFRAIM$Diag_Complex=="UNDERTERMINED","Undertermined",
                                                                                    ifelse(EFRAIM$Diag_Complex=="No_reported_diagnostic_Work_up","No_reported_diagnostic_Work_up","Other"))))))))


EFRAIM$SommeDiag<-EFRAIM[,692]+EFRAIM[,693]+EFRAIM[,694]+EFRAIM[,695]+EFRAIM[,696]+EFRAIM[,696]+
  EFRAIM[,698]+EFRAIM[,699]+EFRAIM[,700]+EFRAIM[,701]+EFRAIM[,702]+
  EFRAIM[,703]+EFRAIM[,704]+EFRAIM[,705]+EFRAIM[,706]+EFRAIM[,707]

EFRAIM$MultipleDiagnoses<-as.factor(ifelse(EFRAIM$SommeDiag<2,"No","Yes"))

EFRAIM$iMV<-as.factor(ifelse(EFRAIM$Comp_risk_IOT=="iMV","Yes","No"))

EFRAIM$PF_Ratio<-c()
EFRAIM$PF_Ratio<-Efraim_3_PF_RATIO$`PaO2/FiO2 VALUE VALUE`
summary(EFRAIM$PF_Ratio)

EFRAIM$PF_Ratio_Per_100<-round(EFRAIM$PF_Ratio/100)
EFRAIM$Center<-as.factor(EFRAIM$Center)

write.csv2(EFRAIM,file="\\\\nas-nup-02.bbs.aphp.fr/SLS-Users/3020127/Bureau/EFRAIM3/EFRAIMstatVDVD.csv")


########################################################################################################################
#################################DESCRIPTIVE ANALYSIS###################################################################
########################################################################################################################



vars<-c("Age","GENDER","SOFA_score","ECOG","Frailty","Charlson_index","Goals_of_care","DelayHosp_ICU","Delay_Sympt_ICU","Location_before_ICU","Hematological_Malignancy","HM_Status","HSCT","Underlying_ImmuneDefect",
        "CART","Weight","Temp","Resp_rate","SpO2","PaO2","PaCO2","PF_Ratio","Neutrophils","Neutropenia","Creat_VD","Initial_Oxygenation_Strategy_2","Vasopressors","RRT","ARDS_Berlin","Coma_CNS","Chemotherapy_ICU",
        "DiagVD","MultipleDiagnoses","Diagnostic_Work_Up","Antibiotic_ICU_adequate","Follow_up_Comp","Comp_risk_IOT","ICU_Mortality","Hospital_Mortality","Day90_Mortality","Hospital_LOS")
catvars<-c("GENDER","ECOG","Goals_of_care","Location_before_ICU","Hematological_Malignancy","HM_Status","HSCT","Underlying_ImmuneDefect",
           "CART","Neutropenia","Initial_Oxygenation_Strategy_2","Vasopressors","RRT","ARDS_Berlin","Coma_CNS","Chemotherapy_ICU",
           "DiagVD","MultipleDiagnoses","Diagnostic_Work_Up","Antibiotic_ICU_adequate","Comp_risk_IOT","ICU_Mortality","Hospital_Mortality","Day90_Mortality")
nonorm<-c("Age","SOFA_score","Frailty","Charlson_index","DelayHosp_ICU","Delay_Sympt_ICU","Weight","Temp","Resp_rate","SpO2","PaO2","PaCO2","PF_Ratio","Neutrophils","Creat_VD",
          "Follow_up_Comp","Hospital_LOS")

vis_miss(EFRAIM[,vars])

tab<-CreateTableOne(vars = vars, data = EFRAIM, factorVars = catvars,includeNA = TRUE)
print(tab,nonnormal=nonorm)


tab1<-CreateTableOne(vars = vars, strata = "Day90_Mortality" , data = EFRAIM, factorVars = catvars,includeNA = TRUE)
print(tab1,nonnormal=nonorm)

tab2<-CreateTableOne(vars = vars, strata = "iMV" , data = EFRAIM, factorVars = catvars)
print(tab2,nonnormal=nonorm)


tab3<-CreateTableOne(vars = vars, strata = "MortalityH" , data = EFRAIM, factorVars = catvars)
print(tab3,nonnormal=nonorm)

library(gtools)
library(lme4)
library(lmerTest)


EFRAIM$NoDiag<-as.factor(ifelse(EFRAIM$Diag_Complex=="UNDERTERMINED", "Undertermined_diag","Other_diag"))
EFRAIM$BAL<-as.factor(ifelse(EFRAIM$BAL_YN==1, "BAL","NoBAL"))


mfit_all <- cuminc(ftime = EFRAIM$Follow_up_Comp, fstatus = EFRAIM$Comp_risk_IOT)
mfit_all

ggcompetingrisks(mfit_all,palette = "JCO",multiple_panels = TRUE,risk.table=TRUE,xlab="Time (days)",tables.theme = theme_cleantable(), title="Cumulative incidence invasive MV while accounting for \n competing risk of discharge alive or mortality",break.time.by=10,pval=TRUE,xlim=c(0,30),ggtheme = theme_bw())

mfit_diag <- cuminc(ftime = EFRAIM$Follow_up_Comp, fstatus = EFRAIM$Comp_risk_IOT,group=EFRAIM$DiagVD)
summary(mfit_diag$Tests)
Fig_IOT_Diag<-ggcompetingrisks(mfit_diag,palette = "JCO",multiple_panels = TRUE,risk.table=TRUE,xlab="Time (days)",tables.theme = theme_cleantable(), title="Cumulative incidence invasive MV while accounting for \n competing risk of discharge alive or mortality",break.time.by=10,pval=TRUE,xlim=c(0,30),ggtheme = theme_bw())

EFRAIM_noniMV<-subset(EFRAIM,Initial_Oxygenation_Strategy!="iMV")
levels(EFRAIM_noniMV$Initial_Oxygenation_Strategy_2)
EFRAIM_noniMV$Initial_Oxygenation_Strategy<-relevel(EFRAIM_noniMV$Initial_Oxygenation_Strategy,ref=2)
levels(EFRAIM$Initial_Oxygenation_Strategy_2)
EFRAIM$Initial_Oxygenation_Strategy_2<-relevel(EFRAIM$Initial_Oxygenation_Strategy_2,ref=4)
EFRAIM_noniMV$Initial_Oxygenation_Strategy_2<-droplevels(EFRAIM_noniMV$Initial_Oxygenation_Strategy_2)
EFRAIM_noniMV$Initial_Oxygenation_Strategy<-droplevels(EFRAIM_noniMV$Initial_Oxygenation_Strategy)
mfit_iMV <- cuminc(ftime = EFRAIM_noniMV$Follow_up_Comp, fstatus = EFRAIM_noniMV$Comp_risk_IOT,group=EFRAIM_noniMV$Initial_Oxygenation_Strategy)
summary(mfit_iMV$Tests)
Fig_IOT_InitialO2<-ggcompetingrisks(mfit_iMV,palette = "JCO",multiple_panels = TRUE,risk.table=TRUE,xlab="Time (days)",tables.theme = theme_cleantable(), title="Cumulative incidence invasive MV according to initial oxygenation modality \n competing risk of discharge alive or mortality",break.time.by=10,pval=TRUE,xlim=c(0,30),ggtheme = theme_bw())
trace(survminer:::ggcompetingrisks.cuminc, edit = T)


EFRAIM$MortalityH<-as.numeric(EFRAIM$Hospital_Mortality)-1
Cox_diag<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD,data=EFRAIM)
km_diag<-survfit(Surv(Hospital_LOS,MortalityH)~ DiagVD,data=EFRAIM)
summary(Cox_diag)
ggsurvplot(km_diag, data=EFRAIM,palette="lancet",conf.int=FALSE,conf.int.alpha=0.4,risk.table=TRUE,tables.theme = theme_cleantable(),legend="none",break.time.by=10,pval=TRUE,title="Cumulative survival",xlim=c(0,30))


Cox_InitialOx<-coxph(Surv(Follow_up_Comp,MortalityH)~ Initial_Oxygenation_Strategy_2,data=EFRAIM)
km_Initialox<-survfit(Surv(Hospital_LOS,MortalityH)~ Initial_Oxygenation_Strategy_2,data=EFRAIM)
summary(Cox_InitialOx)
ggsurvplot(km_Initialox, data=EFRAIM_noniMV,palette="lancet",conf.int=FALSE,conf.int.alpha=0.4,risk.table=TRUE,tables.theme = theme_cleantable(),legend="none",break.time.by=10,pval=TRUE,title="Cumulative survival",xlim=c(0,30))








ggadjustedcurves(test, data= as.data.frame(EFRAIM), variable = "GENDER")

########################################################################################################################
#################################MULTIVARIATE ANALYSIS #################################################################
########################################################################################################################
# Variable of interest : Unknown diag - Mortality hosp (survival), Intubation (survival excluding only iMV day 1) & Day90 (glm)

Cox_diag_m<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                    Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,data=EFRAIM)
summary(Cox_diag_m)

plot(cox.zph(Cox_diag_m))
Cox_diag_m<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Age+Frailty+DelayHosp_ICU+Underlying_ImmuneDefect+
                    Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,data=EFRAIM)
summary(Cox_diag_m)

glm_d90<-glmer(Day90_Mortality~DiagVD+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                 Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_d90, exponentiate=TRUE, conf.int=TRUE),n=21)

EFRAIM_noniMV2<-subset(EFRAIM_noniMV,!is.na(Follow_up_Comp))
EFRAIM_noniMV2<-subset(EFRAIM_noniMV2,!is.na(Comp_risk_IOT))
EFRAIM_noniMV2<-EFRAIM_noniMV2[,-766]
EFRAIM_noniMV2$Initial_Oxygenation_Strategy_2<-droplevels(EFRAIM_noniMV2$Initial_Oxygenation_Strategy_2)
EFRAIM_noniMV2$Initial_Oxygenation_Strategy_2<-relevel(EFRAIM_noniMV2$Initial_Oxygenation_Strategy_2,ref=2)
(EFRAIM_noniMV2$Initial_Ox_Strategy)
Modified_EFRAIM<-finegray(Surv(Follow_up_Comp,Comp_risk_IOT)~.,data=EFRAIM_noniMV2, etype="iMV")
FG_Diag<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Charlson_index+Hematological_Malignancy+
                PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM)
summary(FG_Diag)

# Variable of interest : Initi
Cox_diag_m2<-coxph(Surv(Hospital_LOS,MortalityH)~ Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                    Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,data=EFRAIM)
summary(Cox_diag_m2)

glm_d90_2<-glmer(Day90_Mortality~Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                   Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_d90_2, exponentiate=TRUE, conf.int=TRUE),n=21)


FG_IMV2<-coxph(Surv(fgstart,fgstop,fgstatus)~Initial_Oxygenation_Strategy_2+Charlson_index+Hematological_Malignancy+
                PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM)
summary(FG_IMV2)

FG_IMV2_test<-coxph(Surv(fgstart,fgstop,fgstatus)~Initial_Oxygenation_Strategy_2+Frailty+Hematological_Malignancy+
                 PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM)
summary(FG_IMV2_test)

# Sensitivity analysis Full code 
Cox_diag_m_sen1<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Age+Frailty+DelayHosp_ICU+Underlying_ImmuneDefect+
                    Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,data=EFRAIM)
summary(Cox_diag_m_sen1)


FG_IMV_Sen1<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Charlson_index+Hematological_Malignancy+
                PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,weight=fgwt,data=Modified_EFRAIM)
summary(FG_IMV_Sen1)

glm_d90_Sen1<-glmer(Day90_Mortality~DiagVD+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                 Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_d90_Sen1, exponentiate=TRUE, conf.int=TRUE),n=21)

Cox_diag_m2_Sen1<-coxph(Surv(Hospital_LOS,MortalityH)~ Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                     Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,data=EFRAIM)
summary(Cox_diag_m2_Sen1)

FG_IMV2_Sen1<-coxph(Surv(fgstart,fgstop,fgstatus)~Initial_Oxygenation_Strategy_2+Charlson_index+Hematological_Malignancy+
                 PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,weight=fgwt,data=Modified_EFRAIM)
summary(FG_IMV2_Sen1)

FG_IMV2_Sen1b<-coxph(Surv(fgstart,fgstop,fgstatus)~Initial_Oxygenation_Strategy_2:Full_Code+Charlson_index+Hematological_Malignancy+
                      PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM)
summary(FG_IMV2_Sen1b)

glm_d90_2_Sen1<-glmer(Day90_Mortality~Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                   Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_d90_2_Sen1, exponentiate=TRUE, conf.int=TRUE),n=21)

# Sensitivity analysis MICE
library(mice)
library(lattice)
vars2<-c("Age","GENDER","SOFA_score","ECOG","Frailty","Charlson_index","Goals_of_care","DelayHosp_ICU","Delay_Sympt_ICU","Location_before_ICU","Hematological_Malignancy","HM_Status","HSCT","Underlying_ImmuneDefect",
        "CART","Weight","Temp","Resp_rate","SpO2","PaO2","PaCO2","PF_Ratio","Neutrophils","Neutropenia","Creat_VD","Initial_Oxygenation_Strategy_2","Vasopressors","RRT","ARDS_Berlin","Coma_CNS","Chemotherapy_ICU",
        "DiagVD","MultipleDiagnoses","Diagnostic_Work_Up","Antibiotic_ICU_adequate","Follow_up_Comp","Comp_risk_IOT","ICU_Mortality","Hospital_Mortality","Day90_Mortality","Hospital_LOS","MortalityH")
EFRAIM_PreMice<-EFRAIM[,vars2]
EFRAIM_PreMice<-EFRAIM_PreMice[,-29]
PreMICE<-vis_miss(EFRAIM_PreMice)

imp0 <- mice(EFRAIM_PreMice, maxit = 0)
meth <- imp0$method
meth

EFRAIM_Mice<- mice(EFRAIM_PreMice,m=5,maxit=5,seed=987)



plot(EFRAIM_Mice)

GLMDiagFit <- with(EFRAIM_Mice,glm(Day90_Mortality~DiagVD+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                                       Resp_rate+PF_Ratio+Coma_CNS+Vasopressors+RRT,family="binomial"))
MICE_GLM_DG<-summary(pool(GLMDiagFit))
MICE_GLM_DG$OR<-exp(MICE_GLM_DG$estimate)
MICE_GLM_DG$Lo95CI<-exp(MICE_GLM_DG$estimate-(qnorm(0.975)*MICE_GLM_DG$std.error))
MICE_GLM_DG$Hi95CI<-exp(MICE_GLM_DG$estimate+(qnorm(0.975)*MICE_GLM_DG$std.error))

CoxDiagFit <- with(EFRAIM_Mice,coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Age+Frailty+DelayHosp_ICU+Underlying_ImmuneDefect+
                                       Resp_rate+PF_Ratio+Coma_CNS+Vasopressors+RRT))
MICE_Cox_DG<-summary(pool(CoxDiagFit))
MICE_Cox_DG$HR<-exp(MICE_Cox_DG$estimate)
MICE_Cox_DG$Lo95CI<-exp(MICE_Cox_DG$estimate-(qnorm(0.975)*MICE_Cox_DG$std.error))
MICE_Cox_DG$Hi95CI<-exp(MICE_Cox_DG$estimate+(qnorm(0.975)*MICE_Cox_DG$std.error))


MICE_GLM_fit2 <- with(EFRAIM_Mice,glm(Day90_Mortality~Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                                        Resp_rate+PF_Ratio+Coma_CNS+Vasopressors+RRT,family="binomial"))
MICE_GLM_Ini<-summary(pool(MICE_GLM_fit2))
MICE_GLM_Ini$OR<-exp(MICE_GLM_Ini$estimate)
MICE_GLM_Ini$Lo95CI<-exp(MICE_GLM_Ini$estimate-(qnorm(0.975)*MICE_GLM_Ini$std.error))
MICE_GLM_Ini$Hi95CI<-exp(MICE_GLM_Ini$estimate+(qnorm(0.975)*MICE_GLM_Ini$std.error))

CoxDiagFit2 <- with(EFRAIM_Mice,coxph(Surv(Hospital_LOS,MortalityH)~ Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                                        Resp_rate+PF_Ratio+Coma_CNS+Vasopressors+RRT))
MICE_Cox_Ini<-summary(pool(CoxDiagFit2))
MICE_Cox_Ini$HR<-exp(MICE_Cox_Ini$estimate)
MICE_Cox_Ini$Lo95CI<-exp(MICE_Cox_Ini$estimate-(qnorm(0.975)*MICE_Cox_Ini$std.error))
MICE_Cox_Ini$Hi95CI<-exp(MICE_Cox_Ini$estimate+(qnorm(0.975)*MICE_Cox_Ini$std.error))

kable(MICE_Cox_Ini)


########################################################################################################################
# test model with both diag and O2 (beware of interaction)

Cox_diag_O2_m<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                        Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,data=EFRAIM)
Cox_diag_O2_m2<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                    Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+frailty(Center),data=EFRAIM)
cox_diag_m3<-coxphw(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                                     Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+frailty(Center),data=EFRAIM)
summary(Cox_diag_O2_m)
summary(Cox_diag_O2_m3)

Hazard<-cox.zph(Cox_diag_O2_m)
plot(Hazard)
Hazard<-cox.zph(Cox_diag_O2_m)
plot.cox.zph(Hazard)
ggcoxzph(Hazard,var=c("DiagVD","Initial_Oxygenation_Strategy_2","Age","Frailty"),point.alpha = 0.1)
ggcoxzph(Hazard,var=c("Resp_rate","PF_Ratio_Per_100","DelayHosp_ICU","Underlying_ImmuneDefect"),point.alpha = 0.1)
ggcoxzph(Hazard,var=c("Coma_CNS","Vasopressors","RRT"),point.alpha = 0.1)
ggcoxdiagnostics(Cox_diag_O2_m)

Cox_diag_O2_m<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                       Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT_frailty(Center),data=EFRAIM)
summary(Cox_diag_O2_m)

ggcoxzphFixed(Hazard)
plot.cox.zph(Hazard)
glm_diag_O2_d90<-glmer(Day90_Mortality~DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                 Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_diag_O2_d90, exponentiate=TRUE, conf.int=TRUE),n=24)



Modified_EFRAIM_Diag_O2<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Initial_Oxygenation_Strategy_2+Charlson_index+Hematological_Malignancy+
                 PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM,cluster=Center)
Modified_EFRAIM_Diag_O2_diag<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Initial_Oxygenation_Strategy_2+Charlson_index+Hematological_Malignancy+
                                 PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT,weight=fgwt,data=Modified_EFRAIM)
summary(Modified_EFRAIM_Diag_O2)
Hazard<-cox.zph(Modified_EFRAIM_Diag_O2_diag)
plot(Hazard)
ggcoxzph(Hazard,var=c("DiagVD","Initial_Oxygenation_Strategy_2","Charlson_index","Hematological_Malignancy"),point.alpha = 0.1)
ggcoxzph(Hazard,var=c("Resp_rate","PF_Ratio_Per_100","DelayHosp_ICU","Underlying_ImmuneDefect"),point.alpha = 0.1)
ggcoxzph(Hazard,var=c("Coma_CNS","Vasopressors","RRT"),point.alpha = 0.1)

# test model with both diag and O2 (beware of interaction)

Cox_diag_O2_m_sen1<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                       Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,data=EFRAIM)
summary(Cox_diag_O2_m_sen1)

Cox_diag_O2_m_sen1<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                            Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code+frailty(Center),data=EFRAIM)
summary(Cox_diag_O2_m_sen1)

Cox_diag_O2_m_sen1<-coxph(Surv(Hospital_LOS,MortalityH)~ DiagVD+Initial_Oxygenation_Strategy_2+Age+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
                            Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code,data=EFRAIM)
summary(Cox_diag_O2_m_sen1)

EFRAIM$Age_per_10years<-round(EFRAIM$Age/10)

glm_diag_O2_d90_sen1<-glmer(Day90_Mortality~DiagVD+Initial_Oxygenation_Strategy_2+Age_per_10years+Frailty+Charlson_index+DelayHosp_ICU+Underlying_ImmuneDefect+
+Resp_rate+PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+Full_Code+(1|Center), family="binomial",data=EFRAIM)
print(tidy(glm_diag_O2_d90_sen1, exponentiate=TRUE, conf.int=TRUE),n=25)



Modified_EFRAIM_Diag_O2_sen1<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Initial_Oxygenation_Strategy_2+Charlson_index+Hematological_Malignancy+
                                 PF_Ratio_Per_100+Coma_CNS+Vasopressors+RRT+strata(Center),weight=fgwt,data=Modified_EFRAIM)
summary(Modified_EFRAIM_Diag_O2_sen1)

Modified_EFRAIM_Diag_O2_sen1<-coxph(Surv(fgstart,fgstop,fgstatus)~DiagVD+Initial_Oxygenation_Strategy_2,weight=fgwt,data=Modified_EFRAIM)
summary(Modified_EFRAIM_Diag_O2_sen1)
gof(Modified_EFRAIM_Diag_O2_sen1)


summary(Cox_diag_O2_m)
cox.zph(Cox_diag_test)
plot(cox.zph(Cox_diag_test))
