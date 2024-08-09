""" 
    Re-implementation of Zero Knowledge Clustering Based Adversarial Mitigation in Heterogeneous Federated Learning
    initial split and merge technics as a comparison baseline (chenZeroKnowledgeClustering2021).
"""

from typing import Dict,List
from copy import deepcopy

from clustering.utils import *

from trustfids.clustering.cluster import Cluster,ClusterModel,ClusterXevals, mean_intradistance, mean_interdistance, merge_cluster



class ZeroKnowledgeClustering:
    """chenZeroKnowledgeClustering2021
    Re-implementation of Zero Knowledge Clustering Based Adversarial Mitigation in Heterogeneous Federated Learning
    initial split and merge technics.
    Here it's implementend directly on nodes evaluation instead of models.
    """

    def __init__(
        self,
        xevals: Dict[str, Dict[str, float]],
        participants=[],
        cluster_id=1,
        distance_type: str = "euclidean",
    ):
        """Init ZeroKnowledge clustering parameters.
        Args:
            cluster_id : index to start numbering clusters, default at 1

            local_xevals : Evaluations from participants that are not yet
                in a cluster.

            participants: List of participants to create clusters withn all
                participants have to be in local_xevals, if left empty all
                xevals partcipants are users

            distance_type (str, optional): Distance measurement must take value in ["euclidean","cosin_sim"]. Defaults to "euclidean".

        """
        self.xevals: Dict[str, Dict[str, float]] = xevals
        self.local_xevals: Dict[str, Dict[str, float]] = deepcopy(xevals)
        if participants:
            for key, _ in xevals.items():
                if key not in participants:
                    del self.local_xevals[key]
        self.clusters: List[Cluster] = []
        self.cluster_id: int = cluster_id
        self.distance_type = distance_type

    def create_clusters(self) -> List[List[str]]:
        # Finding participant closest to the evaluations mean.
        mean: Dict[str, float] = mean_xevals(self.local_xevals)
        distance_to_mean: Dict[str, float] = {}
        for p_key, p_evals in self.local_xevals.items():
            distance_to_mean[p_key] = distance_xevals(p_evals, mean, self.distance_type)

        # Closest update to the mean is the first cluster
        c1_key, _ = min(distance_to_mean.items(), key=lambda a: a[1])
        c1 = self.local_xevals[c1_key]
        self.local_xevals.pop(c1_key)

        while self.local_xevals:
            distance_to_c1 = {}
            for p_key, p_evals in self.local_xevals.items():
                distance_to_c1[p_key] = distance_xevals(p_evals, c1, self.distance_type)
            ck_id, k_dist = max(distance_to_c1.items(), key=lambda a: a[1])

            self.clusters.append(self.find_cluster(c1, ck_id))
        self.dynamic_split_merge()
        return [c.participants for c in self.clusters]

    def dynamic_split_merge(self):
        """Split and Merge until threshold
        Hard set rho=0.5 as setted in chenZeroKnowledgeClustering2021
        Return :
            List[Cluster] : A list of clusters that have been splitted and merged
        """
      
        # Rho hardcoded at 0.5 in ZKC
        rho: float = 0.5

        # Compute mean intra_distance
        clusters: List[Cluster] = deepcopy(self.clusters)
        mean_intra = mean_intradistance(clusters)

        #  Dynamic split
        for c in clusters:
            if c.intra >= mean_intra * rho:
                zk = ZeroKnowledgeClustering(
                    self.xevals, cluster_id=self.cluster_id, participants=c.participants
                )
                self.clusters.remove(c)
                splited_clusters = zk.create_clusters()
                self.clusters.append(splited_clusters)
                self.cluster_id = zk.cluster_id

        # Dynamic merge
        mean_inter = mean_interdistance(clusters)
        for c in clusters:
            for c_compared in clusters:
                if c <= mean_inter * rho:
                    merge_cluster(c, c_compared, clusters, self.xevals)
                    self.clusters.remove(c_compared)

    def closest_member_to_center(self, members: List[str] = False) -> str:
        """Return the member clossest to the evaluation mean.
        Args:
            members : subset of participants that should be taken into account.
            Default to false where all participants are take into account.

        """
        # local_xevals subset containing only participants specified in members
        bounded_local_xevals: Dict[str, Dict[str, float]] = {}

        if members:
            # mapping members from local_xevals to bounded_local_xeval
            for member in members:
                bounded_local_xevals[member] = self.local_xevals[member]
        else:
            bounded_local_xevals = self.local_xevals

        mean = mean_xevals(bounded_local_xevals)
        distance_to_mean: Dict[str, float] = {}

        for p_key, p_evals in bounded_local_xevals.items():
            distance_to_mean[p_key] = distance_xevals(p_evals, mean, self.distance_type)
        c1_key, _ = min(distance_to_mean.items(), key=lambda a: a[1])
        return c1_key

    def members_in_bound(self, c1: Dict[str, float], ck_id: str) -> List[str]:
        """Find participants in bound.
        Bound follow definition from chenZeroKnowledgeClustering2021
        Args:
            c1: particpants that have evaluations closest to the center
            ck_id: Center for the new computed cluster.

        """
        bound_b = np.divide(
            distance_xevals(c1, self.local_xevals[ck_id], self.distance_type), 2
        )
        ck_members: List[str] = []
        for p_key, p_evals in self.local_xevals.items():
            if (
                distance_xevals(p_evals, self.local_xevals[ck_id], self.distance_type)
                <= bound_b
            ):
                ck_members.append(p_key)
        return ck_members

    def remove_participants(self, participants: List[str]) -> None:
        """Remove particpants from local_xevals
        Used when participants are affectd to a cluster.
        Args :
            participants: participants that should be removed from local_xevals
        """
        for p in participants:
            self.local_xevals.pop(p)

    def find_cluster(self, c1, ck_id) -> ClusterXevals:
        """Identify a cluster as depicted by chenZeroKnowledgeClustering2021
        Participants placed in the new cluster are also removed from local participants.

        Args :
            c1 : Participant (or evaluations?) that have evaluations closest to the initial mean
            ck_id : Participants that evaluated for the center of the newly created cluster.
        Return :
            Cluster : a cluster object containing participants from the cluster.
        """
        #  Find participants that are in the bound pf ck_id.
        ck_members = self.members_in_bound(c1, ck_id)

        # Find the closest member to the cluster composed of participants in bound
        c_centered = self.closest_member_to_center(ck_members)

        if c_centered == ck_id:
            ck = ClusterXevals(str(self.cluster_id), participants=ck_members, xevals = self.xevals)
            self.cluster_id += 1
            self.remove_participants(ck_members)
            return ck
        else:
            return self.find_cluster(c1, c_centered)
