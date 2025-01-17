package example.akka.remote.server;

import akka.actor.AbstractActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.pf.ReceiveBuilder;

import java.io.IOException;

class Ticker extends AbstractActor {

    private final LoggingAdapter log = Logging.getLogger(context().system(), this);

    // Checks if enough number of deices joined round
    public Ticker() {
        Configuration.ConfigurationDTO configuration = Configuration.get();

        receive(ReceiveBuilder.
                match(Aggregator.CheckReadyToRunLearningMessage.class, s -> {
                    log.info("Received CheckReadyToRunLearningMessage message");
                    log.info("Ticker: numberOfDevices: " + s.participants.size());
                    s.replayTo.tell(new Aggregator.ReadyToRunLearningMessageResponse
                            (s.participants.size() >= configuration.minimumNumberOfDevices), self());
                }).
                matchAny(o -> log.info("received unknown message")).build()
        );
    }
}