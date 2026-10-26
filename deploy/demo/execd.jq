# The sandbox's log lines, in the shape the demo's checks and report read.
#
# execd (run with `--log-format json`) writes one object per request, when the request finishes:
#   {"timestamp":"…Z","level":"INFO","message":"request","kind":"exec|job|stream|callback",
#    "command":"bash","status":200,"outcome":"exit","ms":12}
# and `{"message":"execd listening",…}` once at startup. A callback line also carries `job`,
# `attempt` and, when the attempt failed, `error` (and no `status`).
#
# `execd_events` takes one raw log line (a string) and yields the events it stands for, each
# `{ts_ms, src:"sandbox", event, detail}`; for any line that is not an execd request or startup line it
# yields nothing, so a caller keeps such a line as it is.
#
#   listening           the startup line
#   exec_request        every command or job request answered, stamped when it ARRIVED (end - ms)
#   exec_start          a streamed command, likewise
#   exec_end            a streamed command, stamped when it finished
#   job_ended_callback  one attempt at a job-end callback: job, attempt, status, error, took_ms

def ts_ms:
  capture("^(?<s>[^.Z]+)(\\.(?<f>[0-9]+))?Z$")
  | (((.s + "Z") | fromdateiso8601) * 1000) + (((.f // "0") + "000")[0:3] | tonumber);

# execd logs a command name and a job id with `?`, so in JSON each is a string holding a quoted string.
def unq: if type == "string" and startswith("\"") then (fromjson? // .) else . end;

def sandbox($ts; $event; $detail): {ts_ms: $ts, src: "sandbox", event: $event, detail: $detail};

def execd_events:
  (fromjson? // null) as $l
  | if ($l | type) != "object" or ($l.timestamp | type) != "string" then empty
    elif $l.message == "execd listening" then
      sandbox($l.timestamp | ts_ms; "listening"; {listen: $l.listen, bearer_required: $l.bearer})
    elif $l.message != "request" then empty
    else
      ($l.timestamp | ts_ms) as $end
      | ($end - ($l.ms // 0)) as $start
      | if $l.kind == "callback" then
          sandbox($end; "job_ended_callback";
            {job: ($l.job | unq), attempt: $l.attempt, status: ($l.status // null), error: ($l.error // null), took_ms: $l.ms})
        elif $l.kind == "exec" or $l.kind == "job" or $l.kind == "stream" then
          sandbox($start; "exec_request"; {command: ($l.command | unq)}),
          (if $l.kind == "stream" then
             sandbox($start; "exec_start"; {command: ($l.command | unq)}),
             sandbox($end; "exec_end"; {command: ($l.command | unq)})
           else empty end)
        else empty end
    end;
