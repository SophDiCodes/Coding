package frc.robot.vision;
import edu.wpi.first.wpilibj2.command.SequentialCommandGroup;
import frc.robot.commands.IterativePositioningCommand;

public class AutoPositioningCommand extends SequentialCommandGroup {
    private final AutoPositioningController m_controller;

    public AutoPositioningCommand(AutoPositioningController controller){
        m_controller = controller;

        addCommands(
            m_controller.getAutonomousCommand(),
            new IterativePositioningCommand(m_controller.getDrivetrain(), m_controller.getVision())
        );
    }
}
