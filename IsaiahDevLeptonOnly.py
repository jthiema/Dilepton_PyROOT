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

#Vector of four vector for GenLepton
v_GenLepton = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenLepton",GenLepton)

#Vector of four vector for GenAntiLepton
v_GenAntiLepton = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiLepton",GenAntiLepton)

#Vector of four vector for GenLeptPdgId
v_GenLepPdgId = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenLeptonPdgId",v_GenLeptonPdgId)

#Vector of four vector for GenAntiLeptPdgId
v_GenAntiLepPdgId = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiLepton",v_GenAntiLeptonPdgId)

nUnmatchedleptons = 0

histpt = ROOT.TH2F("Pt for Leptons", "Pt for Leptons",
     150,50.e3,200.e3)

for i in range(10000) :

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
        
    if lepPdgId[0] == genlepPdgId[0] :
        # Fill 
        histpt.Fill(v_lepton.pt(),v_genlepton.pt())
    if lepPdgId[0] == genlepPdgId[1] :
        # Fill 
        histpt.Fill(v_lepton.pt(),v_genlepton.pt())
    else : 
        nUnmatchedleptons += 1
        
        
print('Unmatched Leptons',nUnmatchedleptons)
    
    
    
    
