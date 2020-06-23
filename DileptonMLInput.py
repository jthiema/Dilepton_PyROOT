#!/usr/bin/env python
import ROOT
import math
import numpy

def deltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi > math.pi:
        dphi = dphi - 2*math.pi        
    if dphi < -math.pi:
        dphi = dphi + 2*math.pi        
    return dphi

def deltaR(phi1, eta1, phi2, eta2):
    dphi = deltaPhi(phi1,phi2)
    deta = eta1 - eta2
    return math.sqrt(deta*deta + dphi*dphi) 
    
filename = "/mnt/hadoop/store/group/local/cmstop/jthiema/ntuples2018/production_2018_TAG_V001/production_2018_TAG_V001/ttbarsignalplustau_fromDilepton_0.root"

file = ROOT.TFile.Open(filename, "read")

tree = file.Get("writeNTuple/NTuple")

nEvents = tree.GetEntries()

v_lepPdgId = ROOT.std.vector('int')()
tree.SetBranchAddress("lepPdgId",v_lepPdgId)

v_leptons = ROOT.std.vector('ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>')()
tree.SetBranchAddress("leptons",v_leptons)

v_lepPfIso = ROOT.std.vector('float')()
tree.SetBranchAddress("lepPfIso",v_lepPfIso)

v_lepSCEta = ROOT.std.vector('float')()
tree.SetBranchAddress("lepSCEta",v_lepSCEta)

v_lepID_MuonTight = ROOT.std.vector('int')()
tree.SetBranchAddress("lepID_MuonTight",v_lepID_MuonTight)

v_lepID_ElecCutBased = ROOT.std.vector('int')()
tree.SetBranchAddress("lepID_ElecCutBased",v_lepID_ElecCutBased)

v_lepID_MuonTight = ROOT.std.vector('int')()
tree.SetBranchAddress("lepID_MuonTight",v_lepID_MuonTight)

v_jets = ROOT.std.vector('ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>')()
tree.SetBranchAddress("jets",v_jets)

v_jetPFID = ROOT.std.vector('int')()
tree.SetBranchAddress("jetPFID",v_jetPFID)

v_jetBTagDeepCSV = ROOT.std.vector('float')()
tree.SetBranchAddress("jetBTagDeepCSV",v_jetBTagDeepCSV)

met = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("met",met)

GenTop = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenTop",GenTop)

GenAntiTop = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiTop",GenAntiTop)

GenB = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenB",GenB)

GenAntiB = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiB",GenAntiB)

GenWPlus = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenWPlus",GenWPlus)

GenWMinus = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenWMinus",GenWMinus)


DimuonMLInputList = []
DimuonMLTruthOutputList = []

hGenBdeltaR = ROOT.TH1F("GenBdeltaR", "Delta R between GenB and reco jets", 500, 0, 5)
hGenAntiBdeltaR = ROOT.TH1F("GenAntiBdeltaR", "Delta R between GenAntiB and reco jets", 500, 0, 5)

hGenBdeltaR_gencut = ROOT.TH1F("GenBdeltaR_gencut", "Delta R between GenB and reco jets", 500, 0, 5)
hGenAntiBdeltaR_gencut = ROOT.TH1F("GenAntiBdeltaR_gencut", "Delta R between GenAntiB and reco jets", 500, 0, 5)

for i in range(nEvents):
#for i in range(100000):

#    print "Event "+str(i)

#    DimuonMLInput[]
#    DimuonMLTruthOutput[]

    v_lepPdgId.clear()
    v_leptons.clear()
    v_lepPfIso.clear()
    v_lepSCEta.clear()
    v_lepID_MuonTight.clear()
    v_lepID_ElecCutBased.clear()
    v_lepID_MuonTight.clear()
    v_jets.clear()
    v_jetPFID.clear()
    v_jetBTagDeepCSV.clear()

    tree.GetEntry(i)
    
    # Begin Object Selection

    if len(v_leptons) < 2: continue
    if v_lepPdgId[0]*v_lepPdgId[1] > 0 : continue
    if (v_leptons[0] + v_leptons[1]).M() < 20.0 : continue 
    if (met.Pt() < 20.0) : continue

    if abs(v_lepPdgId[0]) == 11 :
        if v_leptons[0].Pt() < 25 or (abs(v_lepSCEta[0]) > 1.4442 and abs(v_lepSCEta[0]) < 1.566) or v_leptons[0].Eta() > 2.4 : continue
        if v_lepID_ElecCutBased[0] != 4 : continue 

    if abs(v_lepPdgId[1]) == 11 :
        if v_leptons[1].Pt() < 20 or (abs(v_lepSCEta[1]) > 1.4442 and abs(v_lepSCEta[1]) < 1.566) or v_leptons[1].Eta() > 2.4 : continue
        if v_lepID_ElecCutBased[1] != 4 : continue 

    if abs(v_lepPdgId[0]) == 13 :
        if v_leptons[0].Pt() < 25 or v_leptons[0].Eta() > 2.4 : continue
        if v_lepID_MuonTight[0] != 1 or v_lepPfIso[0] > 0.15 : continue 

    if abs(v_lepPdgId[1]) == 13 :
        if v_leptons[1].Pt() < 15 or v_leptons[1].Eta() > 2.4 : continue
        if v_lepID_MuonTight[1] != 1 or v_lepPfIso[1] > 0.15 : continue 


    selectedjets = []
    selectedjetsbtagscore = []

    for j in range(len(v_jets)):
        if v_jets[j].Pt() < 30 or v_jets[j].Eta() > 2.4 : continue 
        if v_jetPFID[j] != 3 : continue 
        if deltaR(v_jets[j].Phi(),v_jets[j].Eta(),v_leptons[0].Phi(),v_leptons[0].Eta()) < 0.4 : continue 
        if deltaR(v_jets[j].Phi(),v_jets[j].Eta(),v_leptons[1].Phi(),v_leptons[1].Eta()) < 0.4 : continue 
        selectedjets.append(v_jets[j])
        selectedjetsbtagscore.append(v_jetBTagDeepCSV[j])


    if len(selectedjets) < 2 : continue 

    selectedbtaggedjets = []

    for j in range(len(selectedjets)):
        if selectedjetsbtagscore[j] < 0.5 : continue
        selectedbtaggedjets.append(selectedjets[j])
        
    if len(selectedbtaggedjets) != 2 : continue 

    for j in range(len(selectedbtaggedjets)):
        if deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()) < deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()) :
            hGenBdeltaR.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()))
        else:
            hGenAntiBdeltaR.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()))

        if abs(GenB.Eta()) < 2.7 and abs(GenAntiB.Eta()) < 2.7 : 
            if deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()) < deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()) :
                hGenBdeltaR_gencut.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()))
            else:
                hGenAntiBdeltaR_gencut.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()))



outHistFile = ROOT.TFile.Open("output.root" ,"RECREATE")
outHistFile.cd()

#c = ROOT.TCanvas("c","c",600,400)

#hGenBdeltaR.Draw()
#c.SaveAs("hGenBdeltaR.png")
hGenBdeltaR.Write()
hGenBdeltaR_gencut.Write()
    
#hGenAntiBdeltaR.Draw()
#c.SaveAs("hGenAntiBdeltaR.png")    
hGenAntiBdeltaR.Write()
hGenAntiBdeltaR_gencut.Write()

    # Fill DimuonMLInput

    # Fill DimuonMLTruthOutput with GenTop, GenAntiTop, GenWPlus, GenWMinus 

#    DimuonMLInputList.append(DimuonMLInput[])
#    DimuonMLTruthOutputList.append(DimuonMLTruthOutput[])
