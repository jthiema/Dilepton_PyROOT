#!/usr/bin/env python
import ROOT
import math
import numpy as np

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

# Beginning RECO branches

# The NTuple TTree is located in the writeNTuple TDirectory for the TFile
tree = file.Get("writeNTuple/NTuple")

# The total number of events (int) in the TTree
nEvents = tree.GetEntries()

# The Lepton PDG ID identifies whether the particle is a electron, positron, muon, or anit-muon
v_lepPdgId = ROOT.std.vector('int')()
tree.SetBranchAddress("lepPdgId",v_lepPdgId)

# Vector of four-vectors for the leptons (electrons and muons)
v_leptons = ROOT.std.vector('ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>')()
tree.SetBranchAddress("leptons",v_leptons)

# Vector of Isolation scores for leptons
v_lepPfIso = ROOT.std.vector('float')()
tree.SetBranchAddress("lepPfIso",v_lepPfIso)

# Super cluster eta of calorimeter activity
v_lepSCEta = ROOT.std.vector('float')()
tree.SetBranchAddress("lepSCEta",v_lepSCEta)

# Integer for the Muon selection
v_lepID_MuonTight = ROOT.std.vector('int')()
tree.SetBranchAddress("lepID_MuonTight",v_lepID_MuonTight)

# Integer for the Electron selection
v_lepID_ElecCutBased = ROOT.std.vector('int')()
tree.SetBranchAddress("lepID_ElecCutBased",v_lepID_ElecCutBased)

# Vector of four-vectors for jets
v_jets = ROOT.std.vector('ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float>>')()
tree.SetBranchAddress("jets",v_jets)

# Integer for jet selection
v_jetPFID = ROOT.std.vector('int')()
tree.SetBranchAddress("jetPFID",v_jetPFID)

# BTag Score for the jet
v_jetBTagDeepCSV = ROOT.std.vector('float')()
tree.SetBranchAddress("jetBTagDeepCSV",v_jetBTagDeepCSV)

# Vector of four-vector for MET
met = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("met",met)

# Beginning GEN branches

# Vector of four-vector for generated top
GenTop = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenTop",GenTop)

# Vector of four-vector for generated anti-top
GenAntiTop = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiTop",GenAntiTop)

# Vector of four-vector for generated b-quark
GenB = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenB",GenB)

# Vector of four-vector for generated anti b-quark
GenAntiB = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiB",GenAntiB)

# Vector of four-vector for generated W Plus
GenWPlus = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenWPlus",GenWPlus)

# Vector of four-vector for generated W Minus
GenWMinus = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenWMinus",GenWMinus)

# Vector of four-vector for generated lepton (GenLepton)

# Vector of four-vector for generated anti-lepton (GenAntiLepton)


DimuonMLInputList = []
DimuonMLTruthOutputList = []

# Decalaration and Booking of TH1 Histograms
hGenBdeltaR = ROOT.TH1F("GenBdeltaR", "Delta R between GenB and reco jets", 500, 0, 5)
hGenAntiBdeltaR = ROOT.TH1F("GenAntiBdeltaR", "Delta R between GenAntiB and reco jets", 500, 0, 5)

hGenBdeltaR_gencut = ROOT.TH1F("GenBdeltaR_gencut", "Delta R between GenB and reco jets", 500, 0, 5)
hGenAntiBdeltaR_gencut = ROOT.TH1F("GenAntiBdeltaR_gencut", "Delta R between GenAntiB and reco jets", 500, 0, 5)

# Decalaration and Booking of TH2 Histograms

input_data = np.empty((0,6,6)) #np input array stack initialization
output_data = np.empty((0,6,3)) #np output array stack initialization
for i in range(nEvents):
#for i in range(10000):

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

    # Very important; this is where the data gets filled from the tree
    tree.GetEntry(i)
    
    # Begin Object Selection

    # Only consider events with at least 2 leptons
    if len(v_leptons) < 2: continue
    # Only consider events with oppositely charged leptons
    if v_lepPdgId[0]*v_lepPdgId[1] > 0 : continue
    # Only consider events with invariant mass of lepton pair > 20
    if (v_leptons[0] + v_leptons[1]).M() < 20.0 : continue 
    # Only consider events with MET > 20
    if (met.Pt() < 20.0) : continue


    # Leading electron/positron cuts
    if abs(v_lepPdgId[0]) == 11 :
        if v_leptons[0].Pt() < 25 or (abs(v_lepSCEta[0]) > 1.4442 and abs(v_lepSCEta[0]) < 1.566) or v_leptons[0].Eta() > 2.4 : continue
        if v_lepID_ElecCutBased[0] != 4 : continue 

    # Subleading electron/positron cuts
    if abs(v_lepPdgId[1]) == 11 :
        if v_leptons[1].Pt() < 20 or (abs(v_lepSCEta[1]) > 1.4442 and abs(v_lepSCEta[1]) < 1.566) or v_leptons[1].Eta() > 2.4 : continue
        if v_lepID_ElecCutBased[1] != 4 : continue 

    # Subleading muon cuts
    if abs(v_lepPdgId[0]) == 13 :
        if v_leptons[0].Pt() < 25 or v_leptons[0].Eta() > 2.4 : continue
        if v_lepID_MuonTight[0] != 1 or v_lepPfIso[0] > 0.15 : continue 

    # Subleading muon cuts
    if abs(v_lepPdgId[1]) == 13 :
        if v_leptons[1].Pt() < 15 or v_leptons[1].Eta() > 2.4 : continue
        if v_lepID_MuonTight[1] != 1 or v_lepPfIso[1] > 0.15 : continue 


    selectedjets = []
    selectedjetsbtagscore = []

    for j in range(len(v_jets)):
        # Jet cuts
        if v_jets[j].Pt() < 30 or v_jets[j].Eta() > 2.4 : continue 
        if v_jetPFID[j] != 3 : continue 
        # Only consider jets that are isolated from leptons
        if deltaR(v_jets[j].Phi(),v_jets[j].Eta(),v_leptons[0].Phi(),v_leptons[0].Eta()) < 0.4 : continue 
        if deltaR(v_jets[j].Phi(),v_jets[j].Eta(),v_leptons[1].Phi(),v_leptons[1].Eta()) < 0.4 : continue 
        selectedjets.append(v_jets[j])
        selectedjetsbtagscore.append(v_jetBTagDeepCSV[j])


    # Only consider events with at least two selected jets
    if len(selectedjets) < 2 : continue 
    energy_jet_list = [jet.E() for jet in selectedjets]
    best_jets=[] #select the four most energetic jets
    # print("lengths: ", len(selectedjets),", ",len(selectedjetsbtagscore))
    for _ in range(4):
        max_idx = energy_jet_list.index(max(energy_jet_list))
        #print("max_idx: ", max_idx)
        best_jets.append([selectedjets[max_idx], selectedjetsbtagscore[max_idx]])
        energy_jet_list.pop(max_idx)
        selectedjets.pop(max_idx)
        selectedjetsbtagscore.pop(max_idx) #pop the max elements to get the next max element
        if len(energy_jet_list) == 0: break #if there are less than 4 selected jets, stop
    input = np.zeros((1,6,6)) #initialization of input for filling
    met_pt = met.Pt()
    met_phi = met.Phi()
    for idx in range(2):
        input[0,:,idx] = np.array([v_leptons[idx].Pt(), v_leptons[idx].Phi(), v_leptons[idx].Eta(), 0, met_pt, met_phi])
    for idx in range(len(best_jets)):
        input[0,:,idx+2] = np.array([best_jets[idx][0].Pt(), best_jets[idx][0].Phi(), best_jets[idx][0].Eta(), best_jets[idx][0].E(), best_jets[idx][0].M(), best_jets[idx][1] ])
    input_data = np.vstack((input_data, input))
    output = np.zeros((1,6,3)) # initialization of output for filling
    output[0,0,:] = np.array([GenB.Pt(), GenB.Phi(), GenB.Eta()])
    output[0,1,:] = np.array([GenAntiB.Pt(), GenAntiB.Phi(), GenAntiB.Eta()])
    output[0,2,:] = np.array([GenWPlus.Pt(), GenWPlus.Phi(), GenWPlus.Eta()])
    output[0,3,:] = np.array([GenWMinus.Pt(), GenWMinus.Phi(), GenWMinus.Eta()])
    output[0,4,:] = np.array([GenTop.Pt(), GenTop.Phi(), GenTop.Eta()])
    output[0,5,:] = np.array([GenAntiTop.Pt(), GenAntiTop.Phi(), GenAntiTop.Eta()])
    output_data = np.vstack((output_data, output))
np.save("X.npy", input_data)
np.save("Y.npy", output_data)


#     selectedbtaggedjets = []

#     for j in range(len(selectedjets)):
#         # Only consider jet with BTag score > 0.5
#         if selectedjetsbtagscore[j] < 0.5 : continue
#         selectedbtaggedjets.append(selectedjets[j])
        
#     # Only consider events with two BTagged jets
#     if len(selectedbtaggedjets) != 2 : continue 


#     # Filling of the Histograms
#     for j in range(len(selectedbtaggedjets)):
#         if deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()) < deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()) :
#             hGenBdeltaR.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()))
#         else:
#             hGenAntiBdeltaR.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()))

#         if abs(GenB.Eta()) < 2.7 and abs(GenAntiB.Eta()) < 2.7 : 
#             if deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()) < deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()) :
#                 hGenBdeltaR_gencut.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenB.Phi(),GenB.Eta()))
#             else:
#                 hGenAntiBdeltaR_gencut.Fill(deltaR(selectedbtaggedjets[j].Phi(),selectedbtaggedjets[j].Eta(),GenAntiB.Phi(),GenAntiB.Eta()))



# outHistFile = ROOT.TFile.Open("output.root" ,"RECREATE")
# outHistFile.cd()

#c = ROOT.TCanvas("c","c",600,400)

# Wrtiting of the histograms

#hGenBdeltaR.Draw()
#c.SaveAs("hGenBdeltaR.png")
# hGenBdeltaR.Write()
# hGenBdeltaR_gencut.Write()
    
#hGenAntiBdeltaR.Draw()
#c.SaveAs("hGenAntiBdeltaR.png")    
# hGenAntiBdeltaR.Write()
# hGenAntiBdeltaR_gencut.Write()

    # Fill DimuonMLInput

    # Fill DimuonMLTruthOutput with GenTop, GenAntiTop, GenWPlus, GenWMinus 

#    DimuonMLInputList.append(DimuonMLInput[])
#    DimuonMLTruthOutputList.append(DimuonMLTruthOutput[])
