__precompile__()

using Base: rand, Filesystem
# using Base.Threads
using Core: Int64, Float64
using ArgParse
using Distributions: Normal
using LinearAlgebra: norm, diag, I, det
using ProgressBars: ProgressBar
# using Pycall

include("mimo.jl")

import .MIMO.par_lib: P_min, P_max, P_inst, R_s, Sigma, alpha_s, zeta, R_c

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--ku"
            help="The antenna number of each node (Minimum is 2)"
            required = true
        "--N", "-n"
            help="The antenna number of each node (Minimum is 2)"
            required = true
        "--ks"
            help="The antenna number of source (Minimum is 2, Ks should not larger than Ku)"
            required = true
        "--ke"
            help="The antenna number of eavesdropper (Minimum is 2)"
            required = true
        "--pu"
            help = "Output power of devices (dBm)"
            required = true
        "--period"
            help = "Simulation period"
            required = true
        end

    return parse_args(s)
end

function dbm2watt(dbm)
    watt = 10^(-3) * 10^(dbm/10)
    return watt
end

function hermitian(mat::Matrix)
    return transpose(conj!(mat))
end

function Output_2d(filename, x, x_range, y)
    fn = filename
    # makefiles if not exist
    if isfile(fn) == false
        # mkpath(directory)
        touch(string(fn))
    end
    # cd(directory)
    io = open(fn, "w")
    for i in 1:x_range
        println(io, string(x[i], " ", y[i]))
    end
    close(io)
    return nothing
end


function main()
    parse_args = parse_commandline()
    K_u = parse(Int64, parse_args["ku"])
    # N = parse(Int64, parse_args["N"])
    K_s = parse(Int64, parse_args["ks"])
    K = min(K_s,K_u)
    simulation_max = parse(Int64, parse_args["period"])
    sigma = dbm2watt(Sigma)
    P_s = P_min:P_inst:P_max
    r_c = R_c
    # println(r_c)
    r_s = R_s

    fullcsi = zeros(length(P_s))
    stbc = zeros(length(P_s))

    for P_s_index in eachindex(P_s)
        
        p_s = dbm2watt(P_s[P_s_index])
        
        # simulation_time = 0

        # fullcsi_counter = 0
        # stbc_counter = 0
        # c_hr_fullcsi = 0

        for simulation_time in ProgressBar(1: simulation_max)

            H_sr = rand(Normal(0,1),(K_s,K_u)) + im * rand(Normal(0,1),(K_s,K_u))

            if K_s >= K_u
                H_sr_2 = hermitian(H_sr) * H_sr
            else
                H_sr_2 = H_sr * hermitian(H_sr)
            end

            c_hr_fullcsi = (zeta 
                * log2(abs(det(
                    Matrix(1.0I, (K,K)) 
                    + 
                    p_s / (K_s * alpha_s * sigma)
                    * H_sr_2
                    ))))
            # println(c_hr_fullcsi)
            if c_hr_fullcsi >= r_s
                fullcsi_counter += 1
            end

            
            # print("\r",
            # "Ps = ", P_s[P_s_index],
            # ", FullCSI = ", c_hr_fullcsi / simulation_time,
            # ", STBC = ", c_hr_stbc / simulation_time,
            # ", Period = ", simulation_time
            # )
        end
        fullcsi[P_s_index] = fullcsi_counter / simulation_max
        # stbc[P_s_index] = c_hr_stbc / simulation_max
        print(
            "Ps = ", P_s[P_s_index],
            ", FullCSI = ", fullcsi[P_s_index],
            ", STBC = ", stbc[P_s_index],
            "\n"
        )
    end

    # file output
    path = string("result_txts/stbc_vs_fullcsi/K_u=",K_u,"/")
    file_fullcsi = string("fullcsi.txt")
    file_stbc = string("stbc.txt")

    if ispath(path) == false
        mkpath(path)
    end
    cd(path)

    Output_2d(file_fullcsi, P_s, length(P_s), fullcsi)
    Output_2d(file_stbc, P_s, length(P_s), stbc)

end

if contains(@__FILE__, PROGRAM_FILE)
    main()
end