package example.akka.remote.shared;

import akka.actor.ActorRef;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Messages {

    public static class StartLearning implements Serializable {
        public String id;

        public StartLearning(String id) {
            this.id = id;
        }
    }

    public static class JoinRoundRequest implements Serializable {
        public LocalDateTime availabilityEndAt;
        public String taskId;
        public String clientId;
        public String address;
        public int port;

        public JoinRoundRequest(LocalDateTime availabilityEndAt, String taskId, String clientId, String address, int port) {
            this.availabilityEndAt = availabilityEndAt;
            this.taskId = taskId;
            this.clientId = clientId;
            this.address = address;
            this.port = port;
        }
    }

    public static class JoinRoundResponse implements Serializable {
        public boolean isLearningAvailable;
        public ActorRef aggregator;

        public JoinRoundResponse(boolean isLearningAvailable, ActorRef aggregator) {
            this.isLearningAvailable = isLearningAvailable;
            this.aggregator = aggregator;
        }
    }


    // P , ??? whats that?
    public static class Sum implements Serializable {
        private int first;
        private int second;

        public Sum(int first, int second) {
            this.first = first;
            this.second = second;
        }

        public int getFirst() {
            return first;
        }

        public int getSecond() {
            return second;
        }
    }

    public static class Result implements Serializable {
        private int result;

        public Result(int result) {
            this.result = result;
        }

        public int getResult() {
            return result;
        }
    }

    // moved to messages, P
    public static class CheckReadyToRunLearningMessage implements Serializable{
        public Map<String, ParticipantData> participants;
        public ActorRef replayTo;
        public CheckReadyToRunLearningMessage(Map<String, ParticipantData> participants, ActorRef replayTo) {
            this.participants = participants;
            this.replayTo = replayTo;
        }
    }

    // moved to messages, P
    public static class RunModule implements Serializable{
        public RunModule(String moduleFileName, String modelConfig, Map<String, Messages.ParticipantData> roundParticipants) {
            this.moduleFileName = moduleFileName;
            this.modelConfig = modelConfig;
            this.roundParticipants = roundParticipants;
        }
        public String moduleFileName;
        public String modelConfig;
        public Map<String, ParticipantData> roundParticipants;


        public Map<String, ParticipantData> getRoundParticipants() {
            return roundParticipants;
        }
    }


    public static class InformAggregatorAboutNewParticipant implements Serializable {
        public ActorRef deviceReference;
        public int port;
        public String clientId;
        public String address;
        public InformAggregatorAboutNewParticipant(ActorRef deviceReference, String clientId, String address, int port) {
            this.deviceReference = deviceReference;
            this.port = port;
            this.clientId = clientId;
            this.address = address;
        }
    }

    public static class StartLearningProcessCommand implements Serializable {
        public String modelConfig;
        public Map<String, Messages.ParticipantData> roundParticipants;
        public StartLearningProcessCommand( String modelConfig, Map<String, ParticipantData> roundParticipants) {
            this.roundParticipants = roundParticipants;
            this.modelConfig = modelConfig;
         }

        public String getModelConfig() {
            return modelConfig;
        }


        public Map<String, ParticipantData> getRoundParticipants() {
            return roundParticipants;
        }
    }

    public static class StartRoundCoordinatorSelector implements Serializable {
        public ActorRef aggregator;

        public StartRoundCoordinatorSelector(ActorRef aggregator) {
            this.aggregator = aggregator;
        }
    }

    public static class StartLearningModule implements Serializable {

        public StartLearningModule() {

        }
    }

    public static class GetModulesListRequest implements Serializable {
        public String id;

        public GetModulesListRequest(String id) {
            this.id = id;
        }
    }

    public static class GetModulesListResponse implements Serializable {
        public List<ModuleData> modules;

        public GetModulesListResponse(List<ModuleData> modules) {
            this.modules = modules;
        }
    }

    public static class GetModuleRequest implements Serializable {
        public String name;

        public GetModuleRequest(String name) {
            this.name = name;
        }
    }

    public static class GetModuleResponse implements Serializable {
        public byte[] content;
        public String fileName;

        public GetModuleResponse(byte[] content, String fileName) {
            this.content = content;
            this.fileName = fileName;
        }
    }

    public static class ModuleData implements Serializable {
        public ModuleData(String id, String fileName, String description, Boolean useCUDA, int minRAMInGB, InstanceType instanceType) {
            this.id = id;
            this.fileName = fileName;
            this.description = description;
            this.useCUDA = useCUDA;
            this.minRAMInGB = minRAMInGB;
            this.instanceType = instanceType;
        }

        public String id;
        public String fileName;
        public String description;

        public Boolean useCUDA;
        public int minRAMInGB;
        public InstanceType instanceType;
    }

    // Stores information about each participant - same for server and client ,P
    public static class ParticipantData {
        public ParticipantData(ActorRef deviceReference, String address, int port) {
            this.deviceReference = deviceReference;
            this.moduleStarted = false;
            this.moduleAlive = false;
            this.port = port;
            this.address = address;
            this.interRes = new ArrayList<>();
        }

        public ActorRef deviceReference;
        public boolean moduleStarted;
        public boolean moduleAlive;
        public int port;
        public String address;
        public List<Float> interRes;
    }

    // not to use????
    public static class ClientDataSpread implements Serializable {
        public ClientDataSpread(String clientId,
                                int numberOfClients,
                                int minimum,
                                Map<String, String> addresses,
                                Map<String, Integer> ports,
                                Map<String, ActorRef> references,
                                List<Float> publicKeys){
            this.clientId = clientId;
            this.numberOfClients = numberOfClients;
            this.minimum = minimum;
            this.addresses = addresses;
            this.ports = ports;
            this.references = references;
            this.publicKeys = publicKeys;
        }

        public String clientId;
        public int numberOfClients;
        public int minimum;
        public Map<String, String> addresses;
        public Map<String, Integer> ports;
        public Map<String, ActorRef> references;
        public List<Float> publicKeys;
    }

    public static class AreYouAliveQuestion implements Serializable { }

    public static class IAmAlive implements Serializable { }

    public static class StartRound implements Serializable { }

    public static class RValuesReady implements Serializable { }

    public static class SendRValue implements Serializable {
        public String sender;
        public byte[] bytes;

        public SendRValue(String sender, byte[] bytes){
            this.sender=sender;
            this.bytes=bytes;
        }
    }

    public static class SendInterRes implements Serializable {
        public String sender;
        public byte[] bytes;

        public SendInterRes(String sender, byte[] bytes){
            this.sender=sender;
            this.bytes=bytes;
        }
    }

    public static class RoundEnded implements Serializable { }

    public enum InstanceType {
        Computer,
        Phone,
    }
}
