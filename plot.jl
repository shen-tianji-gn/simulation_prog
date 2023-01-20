using Base
using Core: String
# using Plots
using Plots
using LaTeXStrings



function file_read(K_u::Int64, type::String)
    """
    Read the data in txt files.
    
    K_u: 2, 4

    type: "fullcsi", "stbc"
    """

    x = []
    y = []
    K_u_set = [2,4]
    type_set = ["fullcsi", "stbc"]
    
    directory = string("result_txts/stbc_vs_fullcsi/K_u=",
        K_u,
        "/",
        type,
        ".txt")

    file_name = open(directory, "r")
    data = readlines(file_name)

    if K_u ∉ K_u_set
        throw(DomainError(K_u, "K_u is not 2 or 4"))
    end

    if type ∉ type_set
        throw(DomainError(type, "type is not fullcsi or stbc"))
    end

    for n in 1:size(data)[1]
        strarray = split(data[n], r"\s+",keepempty=false) 
        numarray = parse.(Float64, strarray)
        append!(x,numarray[1])
        append!(y,numarray[2])
    end

    return x,y
end


x_2_csi,y_2_csi = file_read(2,"fullcsi")
x_4_csi,y_4_csi = file_read(4,"fullcsi")
x_2_stbc,y_2_stbc = file_read(2,"stbc")
x_4_stbc,y_4_stbc = file_read(4,"stbc")


# plot
plot_font = "Computer modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=true)
# scalefontsizes(0)

plot(x_2_csi,y_2_csi, marker = "o", label=string("MIMO, ",L"\mathrm{K_u} = 2"))
plot!(x_4_csi,y_4_csi, label=string("MIMO, ",L"\mathrm{K_u} = 4"))
plot!(xlabel=L"\textrm{Standard text}(r) / \mathrm{cm^3}")
plot!(ylabel="Same font as everything")

