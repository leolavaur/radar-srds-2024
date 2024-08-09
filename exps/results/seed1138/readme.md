### This run is kept for it's specifity
When ran with seed 1138 the distribution `sybils_single_cluster` produce the following results : 
- Client 0, 1 and 2 drop the "Reconaissance" class (which is the poisoning target).
- Client 3 drop "Theft" (which only have a single digit number of samples)

When this apply the following effect applies : 
- Foolsgold discard attackers that are too similar
- Foolsgold discard client 0 to 2 that are too similar 
- So only client 3 data are used for all participants.
- Client 3 data are quite good (as the Theft class is unimportant) 

At the same time, trustfids : 
- Discard attacker due to clustering. 
- Progressively increase the weight of client 3 since it has a better model (up to twice the weight of other participants). 
- But still take into account client 0 to 2 making the end result worst than foolsgold. 

In our eyes this last case is not representative because : 
- Foolsgold discard participants that have valuable informations (but in too little quantity to really matter) : it drop a whole class but is lucky that this class doesn't matter.
- It doesn't seem legit to discard benign participants only because they lacked the same class. 