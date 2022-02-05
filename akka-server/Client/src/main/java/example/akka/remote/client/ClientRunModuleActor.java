package example.akka.remote.client;

import akka.actor.ActorPath;
import akka.actor.ActorRef;
import akka.actor.ActorSelection;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class ClientRunModuleActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    @Override
    public void onReceive(Object message) throws Exception {
        // Message that says to run the module
        if (message instanceof Messages.RunModule) {
            log.info("***** ClientRunModuleActor received 'RunModule' message");
            Messages.RunModule receivedMessage = (Messages.RunModule) message;
            this.runLearning( receivedMessage.moduleFileName,
                    receivedMessage.modelConfig );
        }
    }

    // Runs module
    private void runLearning(String moduleFileName, String modelConfig ) {
        log.info("***** ClientRunModuleActor used 'runLearning' method");
        log.info("---------" + moduleFileName +"/"+ modelConfig);
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            // execute scripts with proper parameters
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            log.info(configuration.pathToModules + moduleFileName);
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToModules + moduleFileName,
                            "--datapath", configuration.datapath,
                            "--data_file_name", configuration.datafilename,
                            "--target_file_name", configuration.targetfilename,
                            "--id", configuration.id,
                            "--host", configuration.host,
                            "--port", String.valueOf(configuration.port),
                            "--data_set_id", String.valueOf(configuration.dataSetId),
                            "--model_config", modelConfig);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
            log.info("Process finished with exitCode "+exitCode);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
