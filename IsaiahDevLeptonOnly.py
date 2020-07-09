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


### Isaiah dev
#Vector of four vector for GenLepton
v_GenLeptons = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenLepton",v_GenLeptons)

#Vector of four vector for GenAntiLepton
v_GenAntiLepton = ROOT.Math.LorentzVector('ROOT::Math::PtEtaPhiM4D<float>')()
tree.SetBranchAddress("GenAntiLepton",v_GenAntiLepton)

#Vector of four vector for GenLeptPdgId
v_GenLepPdgId = ROOT.std.vector('int')()
tree.SetBranchAddress("GenLepton",v_GenLepPdgId)

#Vector of four vector for GenAntiLeptPdgId
v_GenAntiLepPdgId = ROOT.std.vector('int')()
tree.SetBranchAddress("GenAntiLepton",v_GenAntiLepPdgId)
### End of Vectors that needed to be added


nUnmatchedleptons = 0

histpt = ROOT.TH2F("histpt", "Pt for Leptons", 60 , 0 , 1000 , 60 , 0 , 1000)

for i in range(10000) :

    v_lepPdgId.clear()
    v_leptons.clear()
    v_lepPfIso.clear()
    v_lepSCEta.clear()
#    v_lepID_MuonTight.clear()
#    v_lepID_ElecCutBased.clear()
#    v_lepID_MuonTight.clear()
#    v_jets.clear()
#    v_jetPFID.clear()
#    v_jetBTagDeepCSV.clear()
   
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
#    if (met.Pt() < 20.0) : continue
    
    print('Lepton: ',v_lepPdgId)
    print('Gen Lepton: ',v_GenLepPdgId)
    print('\n\n\n')
    
    histpt.Fill(v_lepPdgId[0],v_GenLepPdgId)
    
histpt.SetDirectory(0)
file.Close()
histpt.cd()
histpt.Write()

    
    
    
