"""
Clustering methodologies. 
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from trustfids.clustering.cluster_logger import ClusterLog
from trustfids.clustering.utils import *  # type: ignore


class Cluster(ABC):
    def __init__(
        self,
        id,
        participants: List[str],
        clusters: "Clusters",  # Leverage forward reference to enable type hint in circular reference https://stackoverflow.com/a/33844891
        distance_type: str = "euclidean",
    ):
        """Create a cluster

        Args:
            id (_type_): _description_
            participants (List[str]): _description_
            xevals (Dict[str, Dict[str, float]]): _description_
            distance_type (str, optional): Distance measurement must take value in ["euclidean","cosin_sim"]. Defaults to "euclidean".
            intra (float, optional): _description_. Defaults to None.
        """
        self.id: str = id
        self.participants: List[str] = participants
        if distance_type in ["euclidean", "cosin_sim"]:
            self.distance_type = distance_type
        else:
            raise ValueError(
                f"distance_type must be in ['euclidean','cosin_sim'] but given value is {distance_type}"
            )
        self.inter: Dict[str, float]
        self.intra: float
        self.clusters = clusters

    @abstractmethod
    def cluster_centroid(self) -> Dict[str, float]:
        """Compute the cluster centroid
        Cluster centroid is the mean evaluation on each participants from cluster members
        """
        pass

    @abstractmethod
    def intra_cluster_distance(self) -> float:
        """Intra cluster distance measured as the mean distance to the
        centroid.
        """
        pass

    # Self class in type hints isn't well supported :
    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    # def inter_cluster_distance(self, compared_cluster:Cluster,xevals:
    @abstractmethod
    def inter_cluster_distance(self, compared_cluster) -> float:
        """Update the distance between two clusters
        Distance is measured as the L2norm of the centroids.
        Args:
            c2 : Cluster
        """
        pass

    def add_participants(self, participants: List[str]):
        """Adding new participants to the cluster
        Compute interdistance anew.
        Args :
            participants : List of participants that should be added to the cluster
        """
        self.participants += participants
        self.intra = self.intra_cluster_distance()


class ClusterXevals(Cluster):
    def __init__(
        self,
        id: str,
        participants: List[str],
        xevals: Dict[str, Dict[str, float]],
        clusters: "Clusters",
        distance_type: str = "euclidean",
    ):
        """Create a cluster

        Args:
            id (_type_): _description_
            participants (List[str]): _description_
            xevals (Dict[str, Dict[str, float]]): _description_
            distance_type (str, optional): Distance measurement must take value in ["euclidean","cosin_sim"]. Defaults to "euclidean".
            intra (float, optional): _description_. Defaults to None.
        """
        self.xevals = xevals
        super().__init__(id, participants, clusters, distance_type)
        self.intra = self.intra_cluster_distance()

    def cluster_centroid(self) -> Dict[str, float]:
        """Compute the cluster centroid
        Cluster centroid is the mean evaluation on each participants from cluster members
        """
        centroid = {}
        cluster_size = len(self.participants)
        for participant in self.xevals:
            mean_eval = 0
            for cluster_participant in self.participants:
                mean_eval += self.xevals[cluster_participant][participant]
            centroid[participant] = mean_eval / cluster_size
        return centroid

    def intra_cluster_distance(self) -> float:
        """Intra cluster distance measured as the mean distance to the
        centroid.
        """
        # Compute centroid
        cluster_size = len(self.participants)
        centroid = self.cluster_centroid()

        # Mean distance between all cluster member to the centroid
        mean_distance = 0
        for cluster_member in self.participants:
            mean_distance += distance_xevals(
                self.xevals[cluster_member], centroid, self.distance_type
            )
        mean_distance = mean_distance / cluster_size

        self.intra = mean_distance
        return mean_distance

    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def inter_cluster_distance(self, compared_cluster: Cluster) -> float:
        """Return the interdistance from the current cluster and the one provided as argument

        Args:
            compared_cluster (Cluster): cluster to compute the interdistance from.

        Return:
            float

        Raises:
            ValueError: raised when the current cluster and the observed cluster are the same.
        """
        if compared_cluster.id == self.id:
            raise ValueError(
                "inter_cluster_distance muste be computed on two different clusters"
            )
        return distance_xevals(
            self.cluster_centroid(),
            compared_cluster.cluster_centroid(),
            self.distance_type,
        )


class ClusterModel(Cluster):
    def __init__(
        self,
        id,
        participants: List[str],
        models: Dict[str, NDArray],
        clusters: "Clusters",
        distance_type: str = "euclidean",
    ):
        """Create a cluster

        Args:
            id (_type_): _description_
            participants (List[str]): _description_
            models (Dict[str, NDarray): flattenend model from every participants.
            distance_type (str, optional): Distance measurement must take value in ["euclidean","cosin_sim"]. Defaults to "euclidean".
            intra (float, optional): _description_. Defaults to None.
        """
        self.id: str = id
        self.participants: List[str] = participants
        super().__init__(self.id, participants, clusters, distance_type)

        if distance_type in ["euclidean", "cosin_sim"]:
            self.distance_type = distance_type
        else:
            raise ValueError(
                f"distance_type must be in ['euclidean','cosin_sim'] but given value is {distance_type}"
            )
        self.models = models
        self.intra: float = self.intra_cluster_distance()
        self.inter: Dict[str, float] = {}

    def cluster_centroid(self) -> NDArray:
        """Compute the cluster centroid
        Cluster centroid is the mean evaluation on each participants from cluster members
        """
        return np.mean([self.models[p] for p in self.participants], axis=0)

    def intra_cluster_distance(self) -> float:
        """Intra cluster distance measured as the mean distance to the
        centroid.

        """
        cluster_size = len(self.participants)
        intra_distance: float = 0.0
        if cluster_size == 0:
            return intra_distance

        centroid = self.cluster_centroid()
        for p in self.participants:
            # distance methods comes here.
            intra_distance += distance_models(
                cast(NDArray, self.clusters.data[p]), centroid, self.distance_type
            )
        return intra_distance / cluster_size

    # https://stackoverflow.com/questions/36193540/self-reference-or-forward-reference-of-type-annotations-in-python
    def inter_cluster_distance(self, compared_cluster: "ClusterModel") -> float:
        """Return the interdistance from the current cluster and the one provided as argument

        Args:
            compared_cluster (Cluster): cluster to compute the interdistance from.

        Return:
            float

        Raises:
            ValueError: raised when the current cluster and the observed cluster are the same.
        """
        if compared_cluster.id == self.id:
            raise ValueError(
                "inter_cluster_distance muste be computed on two different clusters"
            )
        return distance_models(
            self.cluster_centroid(),
            compared_cluster.cluster_centroid(),
            self.distance_type,
        )


################################
# Clusters utils
# circular ref when placed into cluster/utils since it manipulate Cluster objects.
# Create an object that contain and manipulate list of clusters ?
################################
class Clusters:
    """Manage the list of cluster and operations that concern this list of cluster."""

    def __init__(
        self, data: (Dict[str, Dict[str, float]] | Dict[str, NDArray]), data_type: str,distance_type:str
    ):
        """_summary_

        Args:
            data (Dict[str, Dict[str, float]] | Dict[str, NDArray]): _description_
            data_type (str): ""
            distance_type (str): distance metric used, e.g. cosin_similarity, euclidean, ...
        """
        if data_type not in ["xevals", "models"]:
            raise ValueError(
                "data_type must be one of the following:", "xevals", "models"
            )
        self.data: Dict[str, Dict[str, float]] | Dict[str, NDArray] = data
        self.clusters: List[Cluster] = []
        self.distance_type = distance_type
        cluster_id: int = 1

        # Init one cluster per participant.
        for participant in data.keys():
            id: str = "c" + str(cluster_id)
            cluster_id += 1
            if data_type == "xevals":
                self.clusters.append(
                    ClusterXevals(
                        id,
                        [participant],
                        cast(Dict[str, Dict[str, float]], self.data),
                        self,
                        distance_type=self.distance_type
                    )
                )
            elif data_type == "models":
                self.clusters.append(
                    ClusterModel(
                        id, [participant], cast(Dict[str, NDArray], self.data), self,distance_type=self.distance_type
                    )
                )
        # Populate self.inter
        self.initial_interdistance()

    def initial_interdistance(self) -> None:
        """Create Clusters inter attribute and set inter-distance for every cluster couple"""
        self.inter: Dict[str, Dict[str, float]] = {c.id: {} for c in self.clusters}
        for cluster in self.clusters:
            for compared_cluster in self.clusters:
                if compared_cluster.id == cluster.id:
                    continue
                self.inter[cluster.id][
                    compared_cluster.id
                ] = cluster.inter_cluster_distance(compared_cluster)

    @property
    def nb_clusters(self) -> int:
        """Number of different clusters

        Returns:
            int:
        """
        return len(self.clusters)

    @property
    def mean_intradistance(self) -> float:
        """Mean intradistance from the current clusters"""
        intra: List[float] = [c.intra for c in self.clusters]
        return float(np.mean(intra))

    @property
    def mean_interdistance(self) -> float:
        """Mean interdistance from the current clusters."""
        return self.interdistance_sum / ((self.nb_clusters**2) - self.nb_clusters)

    @property
    def interdistance_sum(self) -> float:
        """Sum of the interdistance from the current clusters

        Returns:
            float: interdistance sum
        """
        return sum(
            [val for clust_inter in self.inter.values() for val in clust_inter.values()]
        )

    def min_interdistance(self) -> Tuple[Cluster, Cluster, float]:
        """Return the cluster identifier of the two closest cluster with their interdistance

        Returns:
            Tuple[str,str,float]
        """
        couple_list = []

        for c1_key, c1_dict in self.inter.items():
            c2_key, c1_c2_inter = min(c1_dict.items(), key=lambda a: a[1])
            couple_list.append(
                (self.get_cluster(c1_key), self.get_cluster(c2_key), c1_c2_inter)
            )

        return min(couple_list, key=lambda a: a[2])
    
    def get_printable_interdistance(self)->Dict[str,Dict[str,float]]:
        """return interdistance of clusters using cluster participants as 
        key instead of cluster id. 

        Returns:
            Dict[Tuple[str],Dict[Tuple[str],float]]: Interdistance with tuple of participants as key
        """
        printable_clust = {}
        for cid,cluster_inter in self.inter.items(): 
            printable_clust[str(self.get_cluster(cid).participants)] = {str(self.get_cluster(compared).participants):inter for compared,inter in cluster_inter.items()}
        return printable_clust
        
    def get_cluster(self, cluster_id: str)->Cluster:
        """Return the cluster object if the cluster is in clusters list.

        Args:
            cluster_id (str): cluster id.
        """
        for c in self.clusters:
            if c.id == cluster_id:
                return c
        raise IndexError(
            f"{cluster_id} is not in the list of clusters."
        )

    def erase_cluster(self, c: Cluster):
        """Erase all references to c

        Args:
            c (Cluster): _description_
        """
        self.inter.pop(c.id)
        [self.inter[c1].pop(c.id) for c1 in self.inter]
        self.clusters.remove(c)

    def merge_cluster(self, c1: Cluster, c2: Cluster):
        """Merge c1 and c2
        Add all c2 participants into c1, delete c2 from clusters and compute
        new interdistances measure.
        """
        c1.participants += c2.participants
        c1.intra_cluster_distance()
        self.erase_cluster(c2)
        for c in self.clusters:
            if c.id == c1.id:
                continue
            self.inter[c1.id][c.id] = c1.inter_cluster_distance(c)
            self.inter[c.id][c1.id] = self.inter[c1.id][c.id] 
    
def build_clusters(
    data: Dict[str, Dict[str, float]] | Dict[str, NDArray],
    input_type: str = "xevals",
    threshold_type: str = "fixed",
    alpha: float = 1.5,
    distance_type: str = "euclidean",
    reduce_mean:float = 0.25,
    log: bool = False
) -> List[List[str]]:
    """Build clusters from a dictionary of evaluations.

    Cross-evaluations are stored in a dictionary of dictionaries, where the keys
    represent client IDs (cid), and the values are dictionaries of evaluations.
    This structure is comparable to a square matrix, where the rows and columns
    are the clients, and the values are the evaluations.

    Args:
        data: A dictionary of data to cluster, data type depend on the input_type parameter.
        input_type: Define the type of data used to cluster participants must be "xevals" or "models", default to xevals
            "xevals",mean that "data" argument will have the following form :
            {
                "cid1": {
                    "cid1": 0.7,
                    "cid2": 0.6,
                    ...
                },
                "cid2": {
                    "cid1": 0.8,
                    "cid2": 0.5,
                    ...
                },
                ...
            }
            "models" mean that "data" argument will have the following form :
            {
                "cid1": model1,
                "cid2": model2,
                ...
            }
            where model1 and model2 are NDarray representation of the model learnt by participant 1 and 2.
        threshold: Threshold type used to stop hierarchical clustering. Must take one of the following value :
                "fixed" : Alpha is used as the minimum inter-distance threshold for fusion.
                "mean" : Mean cluster interdistance from initial distribution s used for fusion.
                "mean_iterative" :  Mean cluster interdistance used for threshold is re-computed for every cluster fusion.
                "dynamic" : interdistance < alpha * (intradistance(a)+intradistance(b))

        alpha: Coefficient that impact cluster fusion higher alpha leads to
        cluster that are more sparse, default : 2.0
        reduce_mean (float) : experimental parameter used to reduce a threhold type of "mean" or "mean_iterative"
        distance_type : string specifying the distance for cluster creation. Must be in ["euclidean","cosin_sim"]
    Returns:
        A list of clusters, with the following structure:
            [
                ["cid1", "cid2", ...],
                ["cid3", "cid4", ...],
                ...
            ]
        log (bool) : wether the log should be printed in the current file.
    """
    # Threshold calculation can either be :
    # 1. Mean interdistance between clusters
    #       - mean : Computed for one time during clusters creation
    #       - mean_iterative : Computed after each cluster fusion
    # 2. Fixed threshold 
    # 3. dynamic threshold e.g: inter(a,b) <= Alpha x (intra(a)+intra(b))
    if threshold_type not in ["fixed", "mean", "mean_iterative", "dynamic"]:
        raise ValueError(
            "Threshold type must be one of the following:",
            "fixed",
            "mean",
            "mean_iterative",
            "dynamic",
        )
    if input_type not in ["xevals", "models"]:
        raise ValueError(
            "Distance type must be one of the following:", "xevals", "models"
        )
    c = Clusters(data=data, data_type=input_type,distance_type=distance_type)

    # Handling threshold choice
    dynamic = False
    mean_iterative = False
    threshold: float = 0
    if threshold_type == "fixed":
        threshold = alpha
    elif threshold_type == "mean":
        threshold = c.mean_interdistance*reduce_mean
    elif threshold_type == "dynamic":
        dynamic = True
    elif threshold_type == "mean_iterative":
        mean_iterative = True

    logs = ClusterLog(threshold_type,c.get_printable_interdistance())
    # Fusion until threshold is met
    while True:
        c1, c2, min_distance = c.min_interdistance()
        # Handling dynamic threshold
        first_try = False
        if dynamic:
            # Forcing cluster fusion when both cluster have single participant
            if len(c1.participants) == 1 and len(c2.participants) == 1:
                first_try = True
            # dynamic_threshold
            threshold = (c1.intra + c2.intra) * alpha
        if mean_iterative:
            threshold = c.mean_interdistance*reduce_mean

        # Stop condition for hierarchical clustering.
        if (min_distance <= threshold) or first_try:
            logs.log_merge(threshold,min_distance,deepcopy(c1.participants),deepcopy(c2.participants))
            c.merge_cluster(c1, c2)
        else:
            break

        # Case when there is only one cluster left
        if c.nb_clusters == 1:
            break
    if log: 
        logs.print_results([c1.participants for c1 in c.clusters],c.get_printable_interdistance())
        
    return [c1.participants for c1 in c.clusters]
